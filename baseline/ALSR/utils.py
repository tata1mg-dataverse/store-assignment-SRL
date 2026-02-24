import torch
import numpy as np

class Logger:
    def __init__(self,mlflow):
        self.logs = []
        self.mlflow = mlflow
        self.epoch_log_train = {}
        self.epoch_log_val = {}
        self.last_step = -1

    def log(self, key, value, step, batch_type='train'):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        if(batch_type == 'train'):
            if(step not in self.epoch_log_train):
                self.epoch_log_train[step] = []
            self.epoch_log_train[step].append(value)
        else:
            if(step not in self.epoch_log_val):
                self.epoch_log_val[step] = []
            self.epoch_log_val[step].append(value)
        
    def step(self, epoch):
        mean_values_train = np.mean([list(v) for k,v in self.epoch_log_train.items()],axis=0)
        mean_values_val = np.mean([list(v) for k,v in self.epoch_log_val.items()],axis=0)
        metrics_train = ['policy_loss' , 'loss', 'advantages','training_reward', 'avg_cost_train', 'avg_tat_train']
        for i, metr in enumerate(metrics_train):
            self.mlflow.log_metric(metr, mean_values_train[i], step = epoch)    
        metrics_val = ['validation_reward','avg_cost_val','avg_tat_val']
        for i, metr in enumerate(metrics_val):
            self.mlflow.log_metric(metr, mean_values_val[i], step = epoch)    
        self.epoch_log_train = {}
        self.epoch_log_val = {}
        self.last_step = epoch

    def log_raw_val(self, key, value, step):
        self.mlflow.log_metric(key, value, step=step)