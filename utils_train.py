import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#import seg_metrics
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import copy


def train_model(model, dataloaders, criterion, optimizer, sc_plt, writer, device, num_epochs=25):    
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000
    iterations = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            list_dice_val = []

            # Iterate over data.
            for sample in dataloaders[phase]: 
                user_batch = sample[:,0].to(device)
                movie_batch = sample[:,1].to(device)
                labels = sample[:,2].float().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss                    
                    outputs = model([user_batch, movie_batch])
                    
                    # Calculate Loss                    
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        if iterations % 100 == 0:
                            pass
                            #writer.add_scalars('/train/text/outputs', outputs, iterations)
                            #writer.add_scalars('/train/text/labels', labels, iterations)                            
                    
                    
                    # Calculate metric during evaluation
                    if phase == 'val':
                        pass                        

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()                
                if phase == 'train':
                    if iterations % 100 == 0:                        
                        pass
                    iterations += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)            

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            
            writer.add_scalar('epoch/loss_' + phase, epoch_loss, epoch)
            if phase == 'val':
                writer.add_histogram(phase + '/Labels', labels, epoch)
                writer.add_histogram(phase + '/Outputs', outputs, epoch)                    
            
            # Update Scheduler if training loss doesn't change for patience(2) epochs
            if phase == 'train':
                sc_plt.step(epoch_loss)
                
                # Get current learning rate (To display on Tensorboard)
                for param_group in optimizer.param_groups:
                    curr_learning_rate = param_group['lr']
                    writer.add_scalar('epoch/learning_rate_' + phase, curr_learning_rate, epoch)

            # deep copy the model and save if accuracy is better
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())                          

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model