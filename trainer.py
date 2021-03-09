import torch
from evaluator import Evaluator
from torch.utils.data import  DataLoader

class Trainer(object):
    def __init__(self, config, model, criterion, optimizer, save_path):
        self.config = config
        self.loss = criterion
        self.evaluator = Evaluator(loss=self.loss)
        self.optimizer = optimizer
        self.save_path = save_path
        self.device = self.config.device
        self.model = model.to(self.device)
        self.train_loader = None
        self.valid_loader = None

        
    def calculate_accu(self, big_idx, targets):
        n_correct = (big_idx==targets).sum().item()
        return n_correct
    
        
    def train_epoch(self, epoch):
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        self.model.train()
        print('train_epoch', epoch)
        
        for _, batch in enumerate(self.train_loader):
            ids = batch['input_ids'].to(self.device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(self.device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(self.device, dtype=torch.long)
            targets = batch['labels'].to(self.device, dtype=torch.long)
            
            outputs = self.model(ids, attention_mask, token_type_ids, labels=targets)
            loss, logits = outputs[0], outputs[1]
            
            tr_loss += loss.item()
            scores = torch.softmax(logits, dim=-1)
            big_val, big_idx = torch.max(scores.data, dim=-1)
            n_correct += self.calculate_accu(big_idx,targets)
            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
            
            if _ % 1000 == 0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples 
                print(f"Training Loss per 1000 steps: {loss_step}")
                print(f"Training Accuracy per 1000 steps: {accu_step}")
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        epoch_loss = tr_loss/nb_tr_steps
        epoch_accu = (n_correct*100)/nb_tr_examples
        print(f"Training Loss Epoch: {epoch_loss}")
        print(f"Training Accuracy Epoch: {epoch_accu}")
        
        
    def initial_train(self, label_dataset, dev_dataset):
        print('initial train module')
        self.train_loader = DataLoader(label_dataset, **self.config.train_params)
        self.valid_loader = DataLoader(dev_dataset, **self.config.valid_params)

        for epoch in range(self.config.epochs):
            self.train_epoch(epoch)
            dev_loss, dev_acc = self.evaluator.evaluate(self.model, self.valid_loader)
            print()
            
        pass
    
    
    def pseudo_labeling(self):
        pass
    
    
    
    def self_train(self):
        pass
    
    
