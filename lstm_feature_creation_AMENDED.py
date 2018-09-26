import numpy as np
from Bio import SeqIO
import pandas as pd


def di_liang_et_al_features(csv_file = "spreadsheet_of_NNAAIndex_features_for_20_AAs_simplified.csv"):
    df_liang = pd.read_csv(csv_file, sep = ',', index_col = 0, header = None)
    df_liang_header = df_liang.transpose()
    #alphabetical order based on AA name
    AAs = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    df_liang_header.columns = AAs
    di_liang = df_liang_header.to_dict(orient= 'list')
    return di_liang

def dictionary_one_hot():
    AAs = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    di_one_hot = {}
    for (i,aa) in enumerate(AAs):
        di_one_hot[aa] = np.zeros((len(AAs)), dtype = np.float32)
        di_one_hot[aa][i] = 1.0
    return di_one_hot


def dictionary_substitution_matrix_features(filename):
    AAs = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    di_sub_mat_feat = {}
    for line in open(filename):
        if (line[0] in AAs):
            feats = line.split()[1:21] #keep only the first 20 corresponding the "common 20 AAs"
            feats = map(np.float32, feats)
            di_sub_mat_feat[line[0]] = feats
    return di_sub_mat_feat


def aa_element_to_feat_vector(aa_element, di_one_hot, pam30, pam70, blosum80, di_liang):
    one_hot_vec = di_one_hot[aa_element]
    pam30_vec = pam30[aa_element]
    pam70_vec = pam70[aa_element]
    blosum80_vec = blosum80[aa_element]
    liang_vec = np.asarray(di_liang[aa_element], dtype = np.float32)
    output_vector = np.hstack((one_hot_vec, pam30_vec, pam70_vec, blosum80_vec, liang_vec))
    return output_vector, len(output_vector)
        

def fasta_seq_to_numpy_array(seq_record, di_one_hot, pam30, pam70, blosum80, di_liang):
    #may need to unwrap seq_record from biopython...
    seq_length = len(seq_record)
    _, num_feats = aa_element_to_feat_vector('A', di_one_hot, pam30, pam70, blosum80, di_liang)
    #initialize properly sized numpy array
    output_array = np.zeros((num_feats, seq_length), dtype = np.float32)
    for (index, aa_element) in enumerate(seq_record):
        feat_vector, _ = aa_element_to_feat_vector(aa_element, di_one_hot,
                                                   pam30, pam70, blosum80, di_liang)
        output_array[:,index] = feat_vector
    return output_array
    

#Create Numpy array of (num_feats x seq_length) for each element of the two fasta files
#map sequence index to sequence and numpy array using a dictionary--need to keep in same order as RF
#verify that all .fasta sequences in pos and neg files are proteinogenic "20"
def list_of_arrays_for_feats(fasta_file, di_one_hot, pam30, pam70, blosum80, di_liang):
    list_of_arrays = []
    for (index,seq_record) in enumerate(SeqIO.parse(fasta_file, "fasta")):
        current_array = fasta_seq_to_numpy_array(seq_record, di_one_hot,
                                                 pam30, pam70, blosum80, di_liang) 
        list_of_arrays.append(current_array)
    return list_of_arrays


#returns False if non L-amino acids/non-proteinogenic amino acid sequences or True otherwise
def purify_seq(sequence): #takes in seq as string
    output = True
    for char in sequence:
        if (char == 'R' or char == 'H' or char == 'K' or char == 'D' or char == 'E' or char == 'S' or char == 'T' or char == 'N' or char == 'Q' or char == 'C' or char == 'G' or char == 'P' or char == 'A' or char == 'I' or char == 'L' or char == 'M' or char == 'F' or char == 'W' or char == 'Y' or char == 'V'):
            output = True
        else:
            output = False
            break
    return output


#Fix negative instances fasta file to remove X's and other inappropriate amino acids
def purify_instances(fasta_file):
    seq_records_for_writing = []
    for (index, seq_record) in enumerate(SeqIO.parse(fasta_file, "fasta")):
        if purify_seq(seq_record):
            seq_records_for_writing.append(seq_record)
    return seq_records_for_writing


def write_purified_fasta(seq_records_for_writing, output_filename):
    SeqIO.write(seq_records_for_writing, output_filename, "fasta")


def main():
    
    di_liang = di_liang_et_al_features()
    '''
    print(di_liang)
    
    print(di_liang["A"])
    print(di_liang["V"])
    '''
    
    di_one_hot = dictionary_one_hot()
    '''
    print(di_one_hot)
    print(di_one_hot["A"])
    '''
    
    pam30 = dictionary_substitution_matrix_features("/home/myoumans/Thesis_Aim_1/LSTM_Scripts/LSTM_Feature_Creation/PAM30.txt")
    '''
    print(pam30)
    print(pam30['A'])
    print(pam30['V'])
    '''

    
    pam70 = dictionary_substitution_matrix_features("/home/myoumans/Thesis_Aim_1/LSTM_Scripts/LSTM_Feature_Creation/PAM70.txt")
    '''
    print(pam70)
    '''
    blosum80 = dictionary_substitution_matrix_features("/home/myoumans/Thesis_Aim_1/LSTM_Scripts/LSTM_Feature_Creation/BLOSUM80.txt")

    '''
    print(blosum80)
    '''
    alanine_vec, alanine_vec_len = aa_element_to_feat_vector("A", di_one_hot, pam30, pam70, blosum80, di_liang)
    '''
    print(alanine_vec)
    print(alanine_vec_len)
    print(alanine_vec.shape)
    '''

    
if __name__ == "__main__":
    main()

    
