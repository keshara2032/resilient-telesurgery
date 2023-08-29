from .transtcn import *
from .compasstcn import *
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reset_parameters(module):
    if isinstance(module, nn.Linear):
        module.reset_parameters()


def initiate_model(input_dim, output_dim, transformer_params, learning_params, tcn_model_params, model_name):

    d_model, nhead, num_layers, hidden_dim, layer_dim, encoder_params, decoder_params, context = transformer_params.values()

    lr, epochs, weight_decay, patience = learning_params.values()

    if (model_name == 'transformer'):
        print("Creating Transformer")
        model = TransformerModel(input_dim=input_dim, output_dim=output_dim, d_model=d_model, nhead=nhead, num_layers=num_layers,
                                 hidden_dim=hidden_dim, layer_dim=layer_dim, encoder_params=encoder_params, decoder_params=decoder_params)

    elif (model_name == 'tcn'):
        print("Creating TCN")
        model = TCN(input_dim=input_dim, output_dim=output_dim,
                    tcn_model_params=tcn_model_params)

    model = model.cuda()

    # Define the optimizer (Adam optimizer with weight decay)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay, betas=(0.9,0.98), eps=1e-9)


    # Define the learning rate scheduler (ReduceLROnPlateau scheduler)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=patience, verbose=True)

    criterion = nn.CrossEntropyLoss()

    return model, optimizer, scheduler, criterion

# given a tensor of shape [batch,seq_len,features], finds the most common class within the sequence and returns [batch,features]
def find_mostcommon(tensor, device):

    batch_y_bin = torch.mode(tensor, dim=1).values
    batch_y_bin = batch_y_bin.to(device)

    return batch_y_bin

# evaluation loop (supports both window wise and frame wise)
def eval_loop(model, test_dataloader, criterion, dataloader):
    model.eval()
    
    
    with torch.no_grad():
        # eval
        losses = []
        ypreds, gts = [], []
        accuracy = 0
        n_batches = len(test_dataloader)
        
        nypreds,ngts = [],[]
        
        for src, tgt, future_gesture, future_kinematics in test_dataloader:
            
            if(dataloader == "kw"):
                src = src.to(torch.float32)
                src = src.to(device)
                
                tgt = tgt.to(torch.float32)
                tgt = tgt.to(device)  
                
            y = find_mostcommon(tgt, device) #maxpool
            # y = tgt


            y_pred = model(src)  # [64,10]

            # threshold = 0.6
            # by_pred = (y_pred > threshold).int()
            # nypreds.append(by_pred)
            # ngts.append(y)
  
            # ypreds.append(y_pred)
            # gts.append(y)

            # input()

            pred = torch.argmax(y_pred, dim=-1)
            gt = torch.argmax(y, dim=-1)  # maxpool
            # gt = torch.argmax(tgt,dim=-1)

            # print(y_pred.shape, y.shape,pred.shape, gt.shape)
            pred = pred.cpu().numpy()
            gt = gt.cpu().numpy()

            ypreds.append(pred)
            gts.append(gt)

            loss = criterion(y_pred, y) # maxpool
            # loss = criterion(y_pred, y)

            losses.append(loss.item())
            
            accuracy += np.mean(pred == gt)


        # myaccuracy = calc_accuracy(nypreds, ngts)
        # print("My Accuracy:",myaccuracy)
        # ypreds = np.concatenate(ypreds)
        # gts = np.concatenate(gts)

        # # get_classification_report(ypreds,gts,test_dataloader.dataset.get_target_names())

        # # Compare each element and count matches
        # matches = np.sum(ypreds == gts)

        # print('evaluation:', ypreds.shape, gts.shape, matches)

        # # Calculate accuracy
        # accuracy = matches / len(ypreds)
        accuracy = accuracy/n_batches
        print("Accuracy:", accuracy)

        return np.mean(losses), accuracy

# train loop, calls evaluation every epoch
def traintest_loop(train_dataloader, test_dataloader, model, optimizer, scheduler, criterion, epochs, dataloader):


    accuracy = 0
    total_accuracy = []
    # training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for bi, (src, tgt, future_gesture, future_kinematics) in enumerate(tqdm(train_dataloader)):

            optimizer.zero_grad()

            if(dataloader == "kw"):
                src = src.to(torch.float32)
                src = src.to(device)
                
                tgt = tgt.to(torch.float32)
                tgt = tgt.to(device)  
             

            y = find_mostcommon(tgt, device) #maxpool
            # y = tgt
   
            y_pred = model(src)  # [64,10]
            # print('input, prediction, yseq, gt:',src.shape, y_pred.shape,  y.shape, tgt.shape)
            # input()
            
  
            loss = criterion(y_pred, y)  # for maxpool
            # loss = criterion(y_pred, tgt)
            loss.backward()

            running_loss += loss.item()
            
            optimizer.step()
            
        scheduler.step(running_loss)

        print(
            f"Training Epoch {epoch+1}, Training Loss: {running_loss / len(train_dataloader):.6f}")

        # evaluation loop
        val_loss, accuracy = eval_loop(model, test_dataloader, criterion, dataloader)
        print(f"Valdiation Epoch {epoch+1}, Validation Loss: {val_loss:.6f}")

        total_accuracy.append(accuracy)

    return val_loss, accuracy, total_accuracy



def calc_accuracy(pred, gt):
    
    pred = torch.cat(pred, dim=0)
    gt = torch.cat(gt, dim=0)

    correct_predictions = torch.sum(gt == pred)
    total_predictions = gt.numel()  # Total number of elements in the tensor

    accuracy = correct_predictions.item() / total_predictions


    print("Correct predictions:", correct_predictions.item())
    print("Total predictions:", total_predictions)
    print("Accuracy:", accuracy)
    
    return accuracy
