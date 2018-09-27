# lstm_peptides
This code is designed to train a Bidirectional LSTM model on a dataset  
composed of antibacterial peptides and nonantibacterial peptides using  
Python 2.7 and TensorFlow 1.2 (will soon to modified to work with more  
recent versions of TensorFlow without warnings--see Things to note  
below)

## Running the code
In order to use this code:
  1. open file _lstm_tf_optim_tesset_AMENDED.py_ and edit lines 30-32  
  to include directories for saving:  
    1. a summary  
    2. the model itself    
    3. information regarding the model's performance as it trains
  2. call `python lstm_tf_optim_testset_AMENDED.py` from the command  
  line to execute training and store the models in directory mentioned  
  in 2. above.

## Things to note
In more recent versions of TensorFlow such as 1.7 the code currently  
throws 2 Warnings regarding deprecation of certain functions.  I intend  
to fix these deprecation warnings shortly and correct the code to work  
for TensorFlow 1.10.

In addition, despite my attempts to set random seeds, the final training  
results may differ on each run.  Particualarly from what was originally  
reported in the project this code is associated with, but should  
results close to what was reported.


