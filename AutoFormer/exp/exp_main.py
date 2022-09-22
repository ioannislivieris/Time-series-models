import warnings
warnings.filterwarnings('ignore')


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Basic libraries
#
import os
import time
from   tqdm     import tqdm
import numpy    as np


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Models
#
from models import Informer
from models import Autoformer
from models import Transformer


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# PyTorch library
#
import torch
import torch.nn as nn
from   torch import optim


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Visualization library
#
import matplotlib.pyplot as plt


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# User libraries
#
from exp.exp_basic       import Exp_Basic
from utils.tools         import *
from utils.LossFunctions import *



class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        
        model_dict = {
                      'Autoformer':  Autoformer,
                      'Transformer': Transformer,
                      'Informer':    Informer 
                     }
        print('[INFO] Model: ', self.args.model)

        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            print('[INFO] Multi GPU')

        return model

    
    
    def _select_optimizer(self):
        
        if   (self.args.optimizer == 'Adam'):
            model_optim = optim.Adam(self.model.parameters(), lr = self.args.learning_rate)
        elif (self.args.optimizer == 'SGD'):
            model_optim = optim.SGD(self.model.parameters(), lr = self.args.learning_rate, momentum = self.args.momentum)
        elif (self.args.optimizer == 'Adagrad'):
            model_optim = optim.Adagrad(self.model.parameters(), lr = self.args.learning_rate)
        else:
            print('[ERROR] Not known optimizer: {}'.format(self.args.loss))
            print('[INFO] Optimizer = Adam')            
            model_optim = optim.Adam(self.model.parameters(), lr = self.args.learning_rate)
            
        return model_optim

    
    def _select_criterion(self):
        
        if (self.args.loss == 'MSE'):
            criterion = nn.MSELoss()
        elif (self.args.loss == 'MAE'):
            criterion = nn.L1loss()
        elif (self.args.loss == 'MAPE'):
            criterion = MAPELoss
        elif (self.args.loss == 'SMAPE'):
            criterion = SMAPELoss
        else:
            print('[ERROR] Not known loss: {}'.format(self.args.loss))
            print('[INFO] Loss = MSE')
            self.args.loss = 'MSE'
            criterion = nn.MSELoss()
            
            
        return criterion

    
    def evaluation(self, validation_loader, criterion, process):
        
        total_loss = []
        self.model.eval()
        with torch.no_grad():

            # Set Iteration counter
            Iter = 0
            with tqdm(validation_loader, unit="batch") as tepoch:
                for batch_x, batch_y, batch_x_mark, batch_y_mark in tepoch:
                    tepoch.set_description(f"{process}: {Iter+1}/{len(validation_loader)}")
                    
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    #
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    #
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()

                    loss = criterion(pred, true)

                    total_loss.append(loss)
                    
                    # Increate Iteration-counter
                    #
                    Iter += 1
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    
    
    
    
    
    def train(self, setting, train_loader, vali_loader, test_loader):
        
        
        
        # Create directory for model checkpoints
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)


        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience = self.args.patience, 
                                       verbose  = True)

        
        model_optim = self._select_optimizer()
        criterion   = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

            
        # Print training parameters
        #
        print('[INFO] Training epochs: ', self.args.train_epochs)
        print('[INFO] Batch size:      ', self.args.batch_size)
        print('[INFO] Optimizer:       ', self.args.optimizer)
        print('[INFO] Learning rate:   ', self.args.learning_rate)
        print('[INFO] Loss:            ', self.args.loss)
        print('\n')

        
        
        
        # LR scheduler
        #
        if (self.args.lradj == 'Scheduler'):
            scheduler = LRScheduler(optimizer = model_optim, 
                                    patience  = 2*self.args.patience, 
                                    min_lr    = 1e-6, 
                                    factor    = 0.5, 
                                    verbose   = True)
        
        
    
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            
            with tqdm(train_loader, unit="batch") as tepoch:
                for batch_x, batch_y, batch_x_mark, batch_y_mark in tepoch:
                    tepoch.set_description(f"Epoch {epoch+1}/{self.args.train_epochs}")

                
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)

                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())


                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()
                


                # Calculate Loss
                #
                train_loss = np.average(train_loss)
                validation_loss  = self.evaluation(vali_loader, criterion, 'Validation')
                test_loss = self.evaluation(test_loader, criterion, 'Testing')
                
        
                print()        
                print("[INFO] Time: {:.2f}secs".format(time.time() - epoch_time))
                print("[INFO] Epoch: {0} | Training Loss: {1:.5f} - Validation Loss: {2:.5f} - Testing Loss: {3:.5f}".format(
                    epoch + 1, train_loss, validation_loss, test_loss))
                
                
                
                # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                # Early Stopping
                #                
                early_stopping(validation_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                    
                    
                # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                # Learning rate scheduler
                #
                if (self.args.lradj == 'Scheduler'):
                    scheduler( validation_loss )
                else:
                    adjust_learning_rate(model_optim, epoch + 1, self.args)


                print()                    



        best_model_path = path + '/' + 'Model.pth'
        print('[INFO] Path: ', best_model_path )
        self.model.load_state_dict( torch.load(best_model_path) )

        return self.model