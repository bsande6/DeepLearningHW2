# Deep Learning HW2

This repo uses a sequence to sequence network to predict captions for videos using the MLDS dataset.

To create the environment use 

conda env create -f environment.yml 

To run the training code use

 ./model_seq2seq.sh

To test run the command 

 ./hw2_seq2seq.sh MLDS_hw2_1_data/testing_data out.txt

This will test the model using the mlds dataset and write the bleu result to the file out.txtx

