import numpy as np
import torch
import json


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=1e-4, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_ndcg_5 = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_path = save_path
        self.ndcg_path = self.save_path.replace('.pt', '_best_ndcg.pt')

    def __call__(self, val_loss, eval_result, model):

        score = -val_loss
        ndcg_score = eval_result['ndcg_cut_10']
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, self.save_path)
            with open('result/eval_result.json', 'w') as f:
                json.dump(eval_result, f)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, self.save_path)
            with open('result/eval_result.json', 'w') as f:
                json.dump(eval_result, f)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_path):
        """Saves model when validation loss decrease."""
        if 'ndcg' in save_path:
            if self.verbose:
                print(f'Best NDCG@10: {val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), save_path)
        else:
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), save_path)
            self.val_loss_min = val_loss
