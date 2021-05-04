# Classic-Self-Training-for-Few-Shot-Text-Classification
This repo re-implemented the general self-training mechanism by applying a pretrained language model, BERT.

In this, we train the teacher model(BERT) on labeled data to generate pseudo-labels on unlabeled data, train the student model(BERT) on pseudo-labeled and augmented data, and repeat the teacher-student training untill convergence. 

## Install 
    $pip install -r requirements.txt


## Usage
    #Before running this command, recommend to check the argument option through $python train.py --help
    $python train.py --train_path $TRAIN_PATH --test_path $TEST_PATH --save_path $CHECKPOINT --do_augment True
    
