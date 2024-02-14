import comtrade
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os

from tqdm.notebook import tqdm
from tqdm.notebook import trange

import random
import torch
from torch.utils.data import random_split
import seaborn as sns
from sklearn.metrics import f1_score, cohen_kappa_score, balanced_accuracy_score
import torch.nn.functional as F


from collections import defaultdict


def seed_everything(seed: int):
    '''
    This function is used to maintain repeatability
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def target_distribution(*targets, num_classes, comment=False):
    
    ind = np.arange(num_classes)
    width = 0.25
    train_targets, val_targets, test_targets = targets
  
    sns.set(style='white')
    plt.figure(figsize=(7, 4))
    xvals = train_targets.value_counts(normalize=True).sort_index().values[4-num_classes:] * 100 # for Train
    bar1 = plt.bar(ind, xvals, width, color = 'orange', edgecolor='black', zorder=3) 

    yvals = val_targets.value_counts(normalize=True).sort_index().values[4-num_classes:]  * 100 # for Validation
    bar2 = plt.bar(ind+width, yvals, width, color='seagreen', edgecolor='black', zorder=3) 

    zvals = test_targets.value_counts(normalize=True).sort_index().values[4-num_classes:]  * 100 # for Test
    bar3 = plt.bar(ind+2*width, zvals, width, color='olive', edgecolor='black', zorder=3) 

    plt.xlabel("Target label") 
    plt.ylabel('Percentage, %') 
    plt.title(f"Distribution of target labels")
    if comment:
        plt.suptitle(comment)    
    ticks = [*(range(4))]
    plt.xticks(ind+width, ticks[4-num_classes:]) 
    plt.legend((bar1, bar2, bar3), (['Train', 'Validation', 'Test'])) 
    plt.grid(zorder=0)
    plt.show()
    
    
import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))  


## Training function

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=5, SEED=42):
    '''
    Train of a model
    
    Inputs: 
            model      - name of the model to train;
            criterion  - loss function (e.g. nn.CrossEntropyLoss());
            optimizer  - optimization function (e.g. Adam)
            num_epochs - number of epochs for the training process.
    
    Outputs:
            training_stats - dictionary with average CE loss
                             for Train and Validation phase
                             of each epoch   
            model          - model trained     
    '''    
    
    seed_everything(SEED)

    tempdir = 'models/'
    
    if not os.path.exists(tempdir):
        os.mkdir(tempdir)
    best_model_params_path = os.path.join(tempdir, f'best_{model._get_name()}_params.pt')
    torch.save(model.state_dict(), best_model_params_path)

    training_stats = defaultdict(list)    
    best_f1_av = 0
    best_epoch = 0
    
    since = time.time()
    
    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        # Each epoch has a training and validation phase
        for phase in ['Train', 'Valid']:            
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                preds = []
                true_labels = []

            av_loss = []
            

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], f'E{epoch} {phase}\t'):                                           

                inputs = torch.transpose(inputs, 1, 2)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)                    
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
                    else:
                        pred = outputs.argmax(axis=1).cpu().numpy()    
                        labels = labels.cpu().numpy()        
                        preds.extend(pred)
                        true_labels.extend(labels)

                # statistics
                av_loss.append(loss.item())
            
            
            epoch_av_loss = np.mean(av_loss)

            print(f'{phase} average CE loss: {epoch_av_loss:.4f}')
            training_stats[phase] += [epoch_av_loss]
            
            if phase == 'Valid':
                print('Learning rate =', round(optimizer.state_dict()['param_groups'][0]['lr'], 5), '\n')
                scheduler.step()
                score = f1_score(true_labels, preds, average=None, zero_division='warn')
                kappa = cohen_kappa_score(true_labels, preds)
                bal_acc = balanced_accuracy_score(true_labels, preds)
                epoch_f1_av = score.mean()
                print('Metrics:')
                print(f'\t- b_acc = {round(bal_acc, 3)}')
                print(f'\t- kappa = {round(kappa, 3)}')                
                print(f'\t- F1-sc = {np.around(score, 3)}')
                print('---'*10)

            # deep copy the model
            if phase == 'Valid' and epoch_f1_av > best_f1_av:
                best_f1_av = epoch_f1_av
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_params_path)

            print()

    time_elapsed = format_time(time.time() - since)
    print(f'Training complete in {time_elapsed} (hh:mm:ss)')
    print(f'Best average F1-score: {best_f1_av:4f} was at epoch {best_epoch}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    return training_stats, model


## Test function

def test(model, dataloader, device, SEED=42):
    
    '''
    Evaluation of model performance
    
    Inputs: 
            model      - name of the model trained;
            dataloader - loader of the data to test the model on.
    
    Outputs:
            bal_acc    - Balanced accuracy score
            kappa      - Cohen's kappa   
            score      - F1-score metric 
            
    '''
    seed_everything(SEED)

    model.to(device)
    model.eval()

    preds = []
    labels = []

    for inputs, label in tqdm(dataloader):
        inputs = torch.transpose(inputs, 1, 2)
        inputs, label = inputs.to(device), label.to(device)
        with torch.no_grad():
            outputs = model(inputs)            
        pred = outputs.argmax(axis=1).cpu().numpy()    
        label = label.cpu().numpy()        
        preds.extend(pred)
        labels.extend(label)
        
    bal_acc = balanced_accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    score = f1_score(labels, preds, average=None, zero_division='warn')

    return bal_acc, kappa, score


## Function to plot CE losses

def plot_stats(stats, comment=False):
    
    '''
    Plot an average CE loss for
    Train and Validation phase    
    '''
    
    # Use plot styling from seaborn.
    sns.set(style='white')

    # Increase the plot size and font size
    plt.rcParams["figure.figsize"] = (6, 4)

    x = [*range(1, len(stats['Train']) + 1, 1)]
    # Plot the learning curve
    plt.plot(x, stats['Train'], 'b-o', label="Training")
    plt.plot(x ,stats['Valid'], 'g-o', label="Validation")

    # Label the plot
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(x)
    plt.grid()
    if comment:
        plt.suptitle(comment)
    plt.show()