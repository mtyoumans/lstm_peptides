import lstm_splits_and_scaling_wo_small_seqs_AMENDED as ss
import numpy as np
import random
from sklearn.utils import shuffle
from sklearn.metrics import matthews_corrcoef
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMBlockCell
import time
import pickle
import os

##########################################
#Random Seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)

#Parameters--make sure align with validation
num_epochs = 40
batch_size = 128 #Batch size hyperparameter
num_feats = 86 #number of input features per seqeuence element
num_hidden = 512
num_classes = 2

#note 1 is positive antibacterial instance; 0 is negative non-antibacterial instance
learning_rate = 1.0e-4

#######  DIRECTORIES FOR RESULTS--manually change if necessary
summary_directory = "test_set_summary"
model_path = "test_set_model"
file_to_store_curves = "test_set_results"

############################################
#Auxillary Functions
def shuffle_train_and_labels(train_data, train_labels):
    #have to shuffle these together
    shuffled_train_data, shuffled_train_labels = shuffle(train_data, train_labels)
    return shuffled_train_data, shuffled_train_labels


def epoch_batch_creator(train_data, train_labels, batch_size):
    #batch_size is fixed
    N = len(train_data)
    full_batches_possible = N / batch_size
    remainder = N % batch_size #size of last batch
    batches = []
    labels_for_batches = []
    batches_seq_lengths = []
    for batch_index in range(0, full_batches_possible):
        current_list = train_data[(batch_index*batch_size):((batch_index +1)*batch_size)]
        current_labels = train_labels[(batch_index*batch_size):((batch_index +1)*batch_size)]
        seq_lengths = []
        for array in current_list:
            seq_lengths.append(array.shape[1])
        sorted_indices = np.argsort(seq_lengths)[::-1]
        sorted_list = [current_list[i] for i in sorted_indices]
        sorted_labels = [current_labels[i] for i in sorted_indices]
        sorted_lengths = [seq_lengths[i] for i in sorted_indices]
        batches.append(sorted_list)
        labels_for_batches.append(sorted_labels)
        batches_seq_lengths.append(sorted_lengths)
    rem_batch = train_data[(full_batches_possible*batch_size):(full_batches_possible*batch_size+
                                                               remainder+1)]
    rem_labels = train_labels[(full_batches_possible*batch_size):(full_batches_possible*batch_size+
                                                                  remainder+1)]
    rem_seq_lengths = []
    for rem_array in rem_batch:
        rem_seq_lengths.append(rem_array.shape[1])
    rem_batch_si = np.argsort(rem_seq_lengths)[::-1]
    rem_batch_sorted = [rem_batch[i] for i in rem_batch_si]
    rem_labels_sorted = [rem_labels[i] for i in rem_batch_si]
    rem_seq_lengths_sorted = [rem_seq_lengths[i] for i in rem_batch_si]
    batches.append(rem_batch_sorted)
    labels_for_batches.append(rem_labels_sorted)
    batches_seq_lengths.append(rem_seq_lengths_sorted)
    return batches, labels_for_batches, batches_seq_lengths

def batches_to_tensors_array(batches, labels_for_batches):
    tensors_array = []
    seq_lengths_for_tensors_array = []
    for batch in batches: #a batch is num_feats x seq length
        T = batch[0].shape[1] #max seq length in single batch
        current_batch_seq_lengths = []
        batch_tensor = np.zeros((len(batch), T, num_feats), np.float32)
        for (i,array) in enumerate(batch):
            current_array_seq_length = array.shape[1]
            current_batch_seq_lengths.append(current_array_seq_length)
            batch_tensor[i,0:current_array_seq_length,:] = array.transpose()
        tensors_array.append(batch_tensor)
        seq_lengths_for_tensors_array.append(current_batch_seq_lengths)
    tensors_array_labels = labels_for_batches
    return tensors_array, seq_lengths_for_tensors_array, tensors_array_labels


def random_test_set_minibatch(test_data, test_labels, batch_size):
    testset_minibatch_indices = random.sample(range(0, len(test_data)), batch_size)
    minibatch_list = [test_data[i] for i in testset_minibatch_indices]
    minibatch_labels = test_labels[testset_minibatch_indices]
    minibatch_seq_lens = []
    for array in minibatch_list:
        minibatch_seq_lens.append(array.shape[1])
    sorted_indices = np.argsort(minibatch_seq_lens)[::-1]
    minibatch_sorted_list = [minibatch_list[i] for i in sorted_indices]
    minibatch_sorted_labels = [minibatch_labels[i] for i in sorted_indices]
    minibatch_sorted_lengths = [minibatch_seq_lens[i] for i in sorted_indices]

    T = minibatch_sorted_list[0].shape[1]
    minibatch_tensor = np.zeros((len(minibatch_sorted_list), T, num_feats), np.float32)
    for (i, array) in enumerate(minibatch_sorted_list):
        current_seq_len = minibatch_sorted_lengths[i]
        minibatch_tensor[i,0:current_seq_len,:] = array.transpose()
    return minibatch_tensor, minibatch_sorted_lengths, minibatch_sorted_labels

############################################
#Tensor Flow
def BiLSTM_classifier(X, num_hidden, num_classes, seq_lens, fold, istate_fw = None, istate_bw = None, dtype = tf.float32):
    #X is of shape batch_size x seq_lens x num_feats
    #use variable scope to put the variable for each fold in a different namespace
    with tf.variable_scope(str(fold)):
        lstm_fw_cell = LSTMBlockCell(num_hidden)
        lstm_bw_cell = LSTMBlockCell(num_hidden)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                     X, sequence_length = seq_lens, dtype = tf.float32)
        #concatenate both forward and backward outputs fromt the BiLSTM
        #should consist of 1024 outputs if 512 hidden layer size
        X = tf.concat(outputs, axis = 2)
        #take first and last of the 1024 outputs
        #need to find last based on seq lengths
    
        first = X[:,0,:] #batchsize x 1 x 1024
        #last = X[:,-1,:] #may not support; should be batchsize x 1 x 1024;
        last = last_relevant(X, seq_lens) #from https://danijar.com/variable-sequence-lengths-in-tensorflow
        #print("Printing last")
        #print(last)
        #might need function like last_relevant 
        X = tf.concat([first, last] , axis = 1) #batchsize x 1 x 2048
        weights, biases = weight_and_bias(X.get_shape().as_list()[1], num_classes)
        #set the below to work for each element of the batch
        prediction = tf.matmul(X, weights) + biases
        return prediction

def weight_and_bias(input_size, out_size):
    weight = tf.truncated_normal([input_size, out_size], stddev = 0.01)
    biases = tf.constant(0.1, shape = [out_size])
    return tf.Variable(weight), tf.Variable(biases)


#should return only the last output that is relevant in the forward direction due to padding this
#is necessary
def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

#takes in unnormalized preditions and computes cost
def loss_func(prediction, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = prediction)
    return tf.reduce_mean(cross_entropy)

def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    train_op = optimizer.minimize(loss, global_step = global_step)
    return train_op

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(predictions = logits, targets = labels, k = 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def s_metrics(logits, labels, fold): 
    y_pred = tf.argmax(logits, 1)
    s_accuracy, s_acc_update_op = tf.contrib.metrics.streaming_accuracy(y_pred, labels,
                                                                        name = "acc" + str(fold))
    s_tp, s_tp_update_op = tf.contrib.metrics.streaming_true_positives(y_pred, labels,
                                                                       name = "tp" + str(fold))
    s_tn, s_tn_update_op = tf.contrib.metrics.streaming_true_negatives(y_pred, labels,
                                                                       name = "tn" + str(fold))
    s_fp, s_fp_update_op = tf.contrib.metrics.streaming_false_positives(y_pred, labels,
                                                                        name = "fp" + str(fold))
    s_fn, s_fn_update_op = tf.contrib.metrics.streaming_false_negatives(y_pred, labels,
                                                                        name = "fn" + str(fold))
    return s_accuracy, s_acc_update_op, s_tp, s_tp_update_op, s_tn, s_tn_update_op, s_fp, s_fp_update_op, s_fn, s_fn_update_op


def do_eval(sess, eval_correct, data_placeholder, labels_placeholder, seq_lens,
            data, labels, batch_size):
    #batch the data and labels
    batches, labels_for_batches, batches_seq_lengths = epoch_batch_creator(data,
                                                                           labels,
                                                                           batch_size)
    tensors_array, seq_lengths_for_tensors_array, tensors_array_labels = batches_to_tensors_array(batches, labels_for_batches)
        
    #run one epoch of evaluation
    num_examples = 0
    true_count = 0 #accuracy, number correct
    for (i, tensor_array) in enumerate(tensors_array):
        seq_lengths = seq_lengths_for_tensors_array[i]
        current_labels = tensors_array_labels[i]
        #the index may need to be removed
        true_count += sess.run(eval_correct, {data_placeholder: tensor_array, labels_placeholder: current_labels, seq_lens: seq_lengths})
        num_examples += tensor_array.shape[0] #batch size first
    accuracy = float(true_count) / num_examples
    print("  Num examples: %d; Num correct: %d; Accuracy: %0.04f" %(num_examples, true_count, accuracy))
    return accuracy


def do_eval_s_metrics(sess, s_accuracy, s_acc_update_op, s_tp, s_tp_update_op, s_tn, s_tn_update_op, s_fp, s_fp_update_op, s_fn, s_fn_update_op, data_placeholder, labels_placeholder, seq_lens, data, labels, batch_size):
    #batch the data and labels
    batches, labels_for_batches, batches_seq_lengths = epoch_batch_creator(data,
                                                                           labels,
                                                                           batch_size)
    tensors_array, seq_lengths_for_tensors_array, tensors_array_labels = batches_to_tensors_array(batches, labels_for_batches)
    #run one epoch of evaluation
    for (i, tensor_array) in enumerate(tensors_array):
        seq_lengths = seq_lengths_for_tensors_array[i]
        current_labels = tensors_array_labels[i]
        #the index may need to be removed
        sess.run([s_acc_update_op, s_tp_update_op, s_tn_update_op, s_fp_update_op, s_fn_update_op], {data_placeholder: tensor_array, labels_placeholder: current_labels, seq_lens: seq_lengths})
    new_accuracy, true_pos, true_neg, false_pos, false_neg = sess.run([s_accuracy, s_tp, s_tn, s_fp, s_fn])
    print("  New accuracy: {}".format(new_accuracy))
    print("  True positives: {}, True Negatives:  {}, False Positives:  {}, False Negatives:  {}".format(true_pos, true_neg, false_pos, false_neg))
    mcc = (float(true_pos)*float(true_neg) - float(false_pos)*float(false_neg))/(np.sqrt(float(true_pos + false_pos)*float(true_pos + false_neg)*float(true_neg + false_pos)*float(true_neg + false_neg)))
    print("  MCC: {}".format(mcc))
    return new_accuracy, true_pos, true_neg, false_pos, false_neg, mcc

def get_data_modified_for_test():
    ###########################################
    #load lists of numpy arrays as datasets
    #create shuffling, sorting, batching functions
    all_data, labels = ss.load_data("/home/myoumans/Thesis_Aim_1_Amended/LSTM_Amended_Orig/AMENDED_combined_2609_pos_3170_neg_purified_wo_small_seqs_instances.fasta", 2609, 3170)
    final_train_indices, final_test_indices = ss.create_test_train_split(labels)
    final_train_labels = labels[final_train_indices]
    final_test_labels = labels[final_test_indices]
    skf_iterator = ss.cross_validation_folds(final_train_labels)
    final_train_data = [all_data[i] for i in final_train_indices]
    final_test_data = [all_data[i] for i in final_test_indices]
    feature_indices_one_hot = range(0,20) #+ range(80,86) #hard coded

    return skf_iterator, all_data, labels, final_train_indices, final_test_indices, feature_indices_one_hot


def run_training_test_set(batch_size):
    #skf is 5 fold cross validation iterator
    #note first 3 returned will not be used
    skf_iterator, all_data, labels, final_train_indices, final_test_indices, feature_indices_one_hot = get_data_modified_for_test()  

    fold = 6 #representing test set
    
    train_accuracy = []
    test_accuracy = []
    train_mcc = []
    valid_mcc = []

    final_train_data, final_test_data = ss.center_and_scale(all_data, final_train_indices, final_test_indices, feature_indices_one_hot)
    train_labels = labels[final_train_indices]
    test_labels = labels[final_test_indices]

    #build the tesnorflow placeholder variables
    data_placeholder = tf.placeholder(tf.float32, [None, None, num_feats]) #first is none due to batch_size depending on the batch; second is none due to variable length sequences in batches  
    labels_placeholder = tf.placeholder(tf.int32, [None])
    seq_lens = tf.placeholder(tf.int32, [None])

    #build a graph
    predictions = BiLSTM_classifier(data_placeholder, num_hidden, num_classes, seq_lens, fold, istate_fw = None, istate_bw = None, dtype = tf.float32)

    #add loss to graph
    loss = loss_func(predictions, labels_placeholder)

    #add ops that calc and apply gradients
    train_op = training(loss, learning_rate)

    #add the Op to evaluate the model on test and train
    eval_correct = evaluation(predictions, labels_placeholder)
    s_accuracy, s_acc_update_op, s_tp, s_tp_update_op, s_tn, s_tn_update_op, s_fp, s_fp_update_op, s_fn, s_fn_update_op = s_metrics(predictions, labels_placeholder, fold)
    
    #summaries
    summary = tf.summary.merge_all()

    #add the variable initializer Op
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver()

    #create a session
    sess = tf.Session()

    #instantiate a summarywriter to put output summaries and the graph
    
    summary_writer = tf.summary.FileWriter(summary_directory, sess.graph)

    #after everything is built

    #runn the op to initialize the variables
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    #store loss/acc/mcc at every epoch
    train_acc_each_epoch = []
    test_acc_each_epoch = []
    train_mcc_each_epoch = []
    test_mcc_each_epoch = []
    minibatch_train_loss = []
    minibatch_test_loss = []
    #start training
    for epoch in range(num_epochs):
        start_time = time.time()
        #shuffle before epoch batches creation
        final_train_data, train_labels = shuffle_train_and_labels(final_train_data, train_labels)
        
        batches, labels_for_batches, batches_seq_lengths = epoch_batch_creator(final_train_data,
                                                                               train_labels,
                                                                               batch_size)
        tensors_array, seq_lengths_for_tensors_array, tensors_array_labels = batches_to_tensors_array(batches, labels_for_batches)

        for (i, tensor_array) in enumerate(tensors_array):
            seq_lengths = seq_lengths_for_tensors_array[i]
            current_labels = tensors_array_labels[i]
            _, loss_value = sess.run([train_op, loss], {data_placeholder: tensor_array, labels_placeholder: current_labels, seq_lens: seq_lengths})
            minibatch_train_loss.append(loss_value)
            minibatch_tensor, minibatch_lengths, minibatch_labels = random_test_set_minibatch(final_test_data, test_labels, batch_size)
            minibatch_test_loss_value = sess.run(loss, {data_placeholder: minibatch_tensor, labels_placeholder: minibatch_labels, seq_lens: minibatch_lengths})
            minibatch_test_loss.append(minibatch_test_loss_value)
            
            if (i+1) % 16 == 0:
                print("Every 16th Minibatch Batch Train Loss: {}, {}".format(i+1,loss_value))
                print("Every 16th Minibatch Test Loss:        {}, {}".format(i+1,minibatch_test_loss_value))
        duration = time.time() - start_time

        #evaluate on Training Data/valid data for curves
        print("Train and Valid Accuracy Per Epoch:  Epoch number: {}".format(epoch+1))
        current_train_acc = do_eval(sess, eval_correct, data_placeholder, labels_placeholder, seq_lens,
                              final_train_data, train_labels, batch_size)

        current_test_acc = do_eval(sess, eval_correct, data_placeholder, labels_placeholder, seq_lens,
                                   final_test_data, test_labels, batch_size)

        #save the models
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            #update the events file
            summary_str = sess.run(summary, {data_placeholder: tensor_array, labels_placeholder: current_labels, seq_lens: seq_lengths})
            summary_writer.add_summary(summary_str, (epoch+1))
            summary_writer.flush()
            saver.save(sess, os.path.join(model_path, "model_ckpt"), global_step = (epoch + 1))
            
        #training curves
        train_acc_each_epoch.append(current_train_acc)
        test_acc_each_epoch.append(current_test_acc)
        #train_loss_each_epoch.append(loss_value)
    #end of epoch final accuracy and mcc
    accuracy_on_train = do_eval(sess, eval_correct,
                                data_placeholder, labels_placeholder, seq_lens,
                                final_train_data,
                                train_labels, batch_size)

    accuracy_on_test = do_eval(sess, eval_correct, data_placeholder,
                               labels_placeholder, seq_lens,
                               final_test_data,
                               test_labels, batch_size)

    #evaluate using new accuracy and mcc on test set
    print("New version of Test Accuracy, TP, TN, FP, FN, MCC Per Fold :")
    test_accuracy, test_true_pos, test_true_neg, test_false_pos, test_false_neg, test_mcc = do_eval_s_metrics(sess, s_accuracy, s_acc_update_op,
                                                                                                              s_tp, s_tp_update_op, s_tn, s_tn_update_op, s_fp,
                                                                                                              s_fp_update_op, s_fn, s_fn_update_op,
                                                                                                              data_placeholder,
                                                                                                              labels_placeholder, seq_lens,
                                                                                                              final_test_data,
                                                                                                              test_labels, batch_size)

        
    
    
    #close the session
    sess.close()
        
    return accuracy_on_train, accuracy_on_test, train_acc_each_epoch, test_acc_each_epoch, test_accuracy, test_true_pos, test_true_neg, test_false_pos, test_false_neg, test_mcc, minibatch_train_loss, minibatch_test_loss


def main():
    accuracy_on_train, accuracy_on_test, train_acc_each_epoch, test_acc_each_epoch, test_accuracy, test_true_pos, test_true_neg, test_false_pos, test_false_neg, test_mcc, minibatch_train_loss, minibatch_test_loss = run_training_test_set(batch_size)
    print(accuracy_on_train)
    print(accuracy_on_test)
    print("Should agree with above, {}".format(test_accuracy))
    print(test_mcc)
    f = open(file_to_store_curves, 'wb')
    pickle.dump((accuracy_on_train, accuracy_on_test, train_acc_each_epoch, test_acc_each_epoch, test_accuracy, test_true_pos, test_true_neg, test_false_pos, test_false_neg, test_mcc, minibatch_train_loss, minibatch_test_loss), f)
    f.close()
    
    

if __name__ == "__main__":
    main()

