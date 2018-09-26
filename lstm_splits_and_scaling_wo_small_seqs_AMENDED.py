import lstm_feature_creation_AMENDED as fc
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

def load_data(combined_fasta_file, num_pos, num_neg):  #positives are first in file, then negatives
    # use lstm_fc to import arrays of features
    di_liang = fc.di_liang_et_al_features()
    di_one_hot = fc.dictionary_one_hot()
    pam30 = fc.dictionary_substitution_matrix_features("PAM30.txt")
    pam70 = fc.dictionary_substitution_matrix_features("PAM70.txt")
    blosum80 = fc.dictionary_substitution_matrix_features("BLOSUM80.txt")
    
    combined_instances_arrays = fc.list_of_arrays_for_feats(combined_fasta_file, di_one_hot,
                                                            pam30, pam70, blosum80, di_liang)
 
    #Create labels  1 is antibacterial or positive, 0 is non-antibacterial or negative
    pos_labels = np.ones(num_pos, dtype = np.int32)
    neg_labels = np.zeros(num_neg, dtype = np.int32)
    labels = np.concatenate((pos_labels, neg_labels), axis = 0)

    #return all_instances_arrays and labels in same order
    return combined_instances_arrays, labels


def create_test_train_split(labels):
    sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.20, random_state = 0)
    train_index, test_index = next(sss.split(np.zeros(labels.shape[0]), labels))
    return train_index, test_index

#Should only take in labels[train_index]
def cross_validation_folds(train_labels):
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    skf_iterator = skf.split(np.zeros(train_labels.shape[0]),train_labels)
    return skf_iterator


#Run this on each fold and on overall training data/test data before finalizing the score
#Scale each input of the LSTM sequence to unit mean and cetnered on 0 unless that input is part of a one hot vector
#Remember first 20 features are one-hot and maybe last 3 or 8 depending on inclusion of predicted secondary structure
def center_and_scale(data, train_indices, test_indices, feature_indices_one_hot):
    # first find mean and standard deviation of all features in the training set
    mean_of_train, std_of_train = mean_and_std_of_elements_ignores_one_hot(data, train_indices,
                                                                           feature_indices_one_hot)
    train_data = [data[i] for i in train_indices] #since indices are numpy ndarrays
    test_data = [data[i] for i in test_indices]
    scaled_train_data = standardize(train_data, mean_of_train, std_of_train,
                                    feature_indices_one_hot)
    scaled_test_data = standardize(test_data, mean_of_train, std_of_train,
                                   feature_indices_one_hot)
    return scaled_train_data, scaled_test_data
    


def mean_and_std_of_elements_ignores_one_hot(data, train_indices, feature_indices_one_hot):
    N = 0 #sum of number of elements in all sequences
    num_feats = data[0].shape[0]
    sum_of_elements_feats = np.zeros((num_feats,1), dtype = np.float32)
    train_data = [data[i] for i in train_indices] #since indices are numpy ndarrays
    for train_sample in train_data:
        N += train_sample.shape[1] #number of columns/elements (note features are rows, seq len is columns)
        sum_of_elements_feats += np.sum(train_sample, axis = 1, keepdims = True)   
    mean_of_train = sum_of_elements_feats / N
    #replace mean_of_train with 0 where one_hot is present
    mean_of_train[feature_indices_one_hot,0] = 0.0
    sum_of_squared_elements_feats_minus_mean = np.zeros((num_feats,1), dtype = np.float32)
    for train_sample in train_data:
        sum_of_squared_elements_feats_minus_mean += np.sum(np.square(train_sample
                                                                     - mean_of_train),
                                                           axis = 1, keepdims = True)
    std_of_train = np.sqrt((sum_of_squared_elements_feats_minus_mean / (N-1)))#estimate of sigma
    std_of_train[feature_indices_one_hot,0] = 1.0
    std_of_train[range(80,86),0] = 1.0  #hard coded so just mean center the physicochemical features
    return mean_of_train, std_of_train


def standardize(data, mean_of_train_data, std_of_train_data, feature_indices_one_hot):
    #Use just the train_data to scale to zero mean unit variance for each element of each sequence on average
    indices = feature_indices_one_hot
    output_data = []
    for (i,sample) in enumerate(data):
        new_sample = (sample - mean_of_train_data)/ std_of_train_data
        #may not work if indices not contiguous; test this
        output_data.append(new_sample)
    return output_data



def main():
    all_instances_arrays, labels = load_data()
    train_index, test_index = create_test_train_split(labels)
    print("Train_index, test_index below")
    print((train_index, test_index))
    train_labels = labels[train_index]
    #print(train_labels)
    skf_iterator = cross_validation_folds(train_labels)
    
    data = all_instances_arrays
    feature_indices_one_hot = range(0,20) #+ range(80,86)
    scaled_train_data, scaled_test_data = center_and_scale(data, train_index, test_index, feature_indices_one_hot)    
    #print(feature_indices_one_hot)
    print(scaled_train_data[0][:,0])

    mean_of_train, std_of_train = mean_and_std_of_elements_ignores_one_hot(data, train_index, feature_indices_one_hot)
    print(mean_of_train)
    print(std_of_train)

    
if __name__ == "__main__":
    main()
