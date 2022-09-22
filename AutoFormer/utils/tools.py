import numpy as np
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
# plt.switch_backend('agg')


import torch
import random

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('[INFO] Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'[INFO] EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'[INFO] Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
        torch.save(model.state_dict(), path + '/' + 'Model.pth')
        self.val_loss_min = val_loss


        
class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5, verbose = True):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience  = patience
        self.min_lr    = min_lr
        self.factor    = factor
        self.verbose   = verbose
        
#         self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = self.optimizer, 
#                                                                        steps     = self.patience, 
#                                                                        verbose   = self.verbose )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode      = 'min',
                patience  = self.patience,
                factor    = self.factor,
                min_lr    = self.min_lr,
                verbose   = self.verbose 
            )
        
    def __call__(self, val_loss):
        self.lr_scheduler.step( val_loss )
        
        
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



def visual(inputs, true, preds, figsize = (15, 3), name='test.pdf'):
    """
    Results visualization
    """

    Lag     = inputs.shape[1]
    Horizon = true.shape[1]
    
    subplots = [421, 422, 423, 424, 425, 426, 427, 428]
    plt.figure( figsize = (30, 10) )
    RandomInstances = [random.randint(1, 12) for i in range(0, 8)]

    for plot_id, i in enumerate(RandomInstances):

        plt.subplot(subplots[plot_id]);
        plt.grid();
        plt.plot(       np.arange(0, Lag),     inputs[i], color='tab:blue', marker = 'o');
        plt.plot( Lag + np.arange(0, Horizon), true[i],   color='g', marker = 'o');
        plt.plot( Lag + np.arange(0, Horizon), preds[i],  color='r', marker = 'o');

        plt.legend(['Input', 'Ground truth', 'Prediction', ], frameon = False, fontsize = 12);
        plt.savefig(name, bbox_inches='tight');
    plt.close();

