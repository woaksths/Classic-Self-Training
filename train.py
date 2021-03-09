import argparse
import constant as config
import torch
from dataset import read_dataset, sampling_dataset, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import  DataLoader
from trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', required = True)
parser.add_argument('--test_path', required = True)
parser.add_argument('--save_path', required = True)
args = parser.parse_args()

# Read dataset
train_texts, train_labels = read_dataset(args.train_path, config.class_num)
test_texts, test_labels = read_dataset(args.test_path, config.class_num)

# Dataset sampling -> Since we are going to simulate semi-supervised learning, we will assume that we only know a little part of labeled data.
labeled_data, remained_data = sampling_dataset(list(zip(train_texts, train_labels)), class_num=config.class_num, sample_num=30)
dev_data, unlabeled_data = sampling_dataset(remained_data, class_num=config.class_num, sample_num=30)

# Tokenizing 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

labeled_texts = [data[0] for data in labeled_data]
labeled_labels = [data[1] for data in labeled_data]
labeled_encodings = tokenizer(labeled_texts, truncation=True, padding=True)
label_dataset = Dataset(labeled_encodings, labeled_labels)

dev_texts = [data[0] for data in dev_data]
dev_labels = [data[1] for data in dev_data]
dev_encodings = tokenizer(dev_texts, truncation=True, padding=True)
dev_dataset = Dataset(dev_encodings, dev_labels)

# We keep the label of unlabeled data to track for accuracy of pseudo-labeling
unlabeled_texts = [data[0] for data in unlabeled_data]
unlabeled_labels = [data[1] for data in unlabeled_data] 

# # Load dataset
# train_loader = DataLoader(label_dataset, **config.train_params)
# valid_loader = DataLoader(dev_dataset, **config.valid_params)

# Build model 
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=config.class_num)
print(model)

# Criterion & optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5) #or AdamW

# Init Trainer
trainer = Trainer(config, model, loss_function, optimizer, args.save_path)

trainer.initial_train(label_dataset, dev_dataset)
1/0
# trainer.self_train()

# Train model 

# Evaluation
