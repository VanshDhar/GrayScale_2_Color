import torch
import os
from colorize_data import ColorizeData
from basic_model import Net
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import math
import sys

class Trainer:
    def __init__(self, val_usage,dataset_path,learning_rate = 2e-3,batch_size = 32,epochs = 100):
        # Define hparams here or load them from a config file
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_usage = val_usage
        self.dataset_path = dataset_path
        if(val_usage=='with'): self.train_test_split=0.15
        else: self.train_test_split=0.0
        
    
        
    def train(self):
        pass
        # dataloaders
        train_dataset = ColorizeData(img_dir=self.dataset_path, train='train',train_val_split=self.train_test_split)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        if(val_usage=='with'):
            val_dataset = ColorizeData(img_dir=self.dataset_path, train='val',train_val_split=self.train_test_split)
            val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)#self.batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        # Model
        model = Net().to(self.device)
        # Loss function to use
        criterion = nn.MSELoss()
        # You may also use a combination of more than one loss function 
        # or create your own.
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)#torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)#torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=0.75)
        # train loop
        
        path = './saved_model'

        if not os.path.exists(path):
          os.makedirs(path)
        
        size = len(train_dataloader.dataset)
        for epoch in range(1,self.epochs+1): 
            model.train()
            for batch, (X, y) in enumerate(train_dataloader):
                
                # Compute prediction and loss
                if self.device == 'cuda':
                  X = X.cuda()
                  y = y.cuda()
                pred = model(X)
                loss = criterion(pred, y)
        
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                if batch % 10 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"epoch {epoch} : loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                    
            torch.save(model, './saved_model/nn_e'+(str(epoch)).zfill(3)+'.pt')
            if(self.val_usage=='with'): self.validate(model,val_dataloader,criterion)
            scheduler.step()
            
            
        
    def validate(self,model,val_dataloader,criterion):
        #pass
        # Validation loop begin
        size = len(val_dataloader.dataset)
        num_batches = len(val_dataloader)
        test_loss = 0
        total_pred = []
        total_y = []
        model.eval()
        with torch.no_grad():
            for X, y in val_dataloader:
                if self.device == 'cuda':
                  X = X.cuda()
                  y = y.cuda()
                pred = model(X)
                total_y.append(y)
                total_pred.append(pred)
                test_loss += criterion(pred, y).item()
    
        test_loss /= num_batches
        print(f"Test Error:  Avg loss: {test_loss:>8f} \n")
        # Validation loop end
        # ------
        # Determine your evaluation metrics on the validation dataset.
        #cosine similarity and scaled euclidean distance
        
        #MEAN = 255 * torch.tensor([0.5, 0.5, 0.5],device=self.device)
        #STD = 255 * torch.tensor([0.5, 0.5, 0.5], device=self.device)
        
        avg_cosine_similarity = 0
        avg_scaled_edistance = 0
        cos = nn.CosineSimilarity(dim=0)
        for y,pred in zip(total_y,total_pred):
          
          y = torch.flatten(y)
          pred = torch.flatten(pred)
          y = y.mul_(255*0.5).add_(255*0.5)
          pred = pred.mul_(255*0.5).add_(255*0.5)
          edist_scaled = math.sqrt(((y-pred)**2).sum(axis=0)) / (255*256*math.sqrt(3))
          avg_scaled_edistance+= edist_scaled
          avg_cosine_similarity+= float(cos(y,pred))
        
        #print(num_batches)
        print('Avg Cosine Similarity on Validation Data: ',avg_cosine_similarity/size)
        print('Avg Scaled Eucledian Distance on Validation Data: ',avg_scaled_edistance/size,'\n')
        #edist = ((A-B)**2).sum(axis=0)


    
        

if __name__=="__main__":
    
    val_usage = sys.argv[1]
    dataset_path = sys.argv[2]
    
    
    if( val_usage!='with') and (val_usage!='without'):
        print("Wrong Input for Validation Usage")
    elif not os.path.exists(dataset_path):
        print("Dataset path doesn't exists")
    else:
        ct = Trainer(val_usage,dataset_path)
        ct.train()
    
