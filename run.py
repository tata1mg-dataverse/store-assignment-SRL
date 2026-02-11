import numpy as np
import pandas as pd 
import copy
import torch 
from tqdm import tqdm

from config import get_config
from SupervisedTrainer import  Trainer
from models import create_policy_model, create_value_model
from input_generator import DataPreparator
from reward_generator import RewardCalculator


def run(config):
    
    path = config.path
    orders = pd.read_csv(f'{path}/orders.csv')
    order_skus = pd.read_csv(f'{path}/order_skus.csv')

    df = order_skus.merge(orders,on=['GROUP_ID','ORDER_ID'])
    df['ORDER_DATE'] = pd.to_datetime(df['ORDER_DATE'],format='mixed')
    df['IS_fast_delivery'] = df['DELIVERY_TYPE'].map({'fast_delivery':1,'slow_delivery':0})

    df['COLD_STORAGE'] = 0
    df['ITEM_WEIGHT_GMS'] = 100
    df['real_actions'] = df['WAREHOUSE_CODE']

    preparator = DataPreparator(path)
    df = df[df.CENTRAL_PINCODE!=-1]
    df = preparator.prepare_data(df)
    df = df[df.inputs.map(len)==149]
    
    training_batches = preparator.create_batches(df[df.ORDER_DATE<'2024-05-10'],149, num_of_batch=80 ,device = config.device)
    validation_batches = preparator.create_batches(df[df.ORDER_DATE>='2024-05-10'],149, num_of_batch=20, device = config.device)


    class_weights = df.shape[0] / (df.WAREHOUSE_CODE.nunique() * np.bincount(df['real_actions']))

    reward_generator = RewardCalculator(path, config.alpha, config.fast_delivery_alpha)

    reward_generator.class_weights = class_weights

    reward_generator.class_weights[reward_generator.class_weights==np.inf] = 0
    reward_generator.class_weights = torch.tensor(reward_generator.class_weights).to(config.device)

    config.reward_generator = reward_generator

    reward_generator.num_scalar_features = training_batches[0]['input'].shape[2]
    print(reward_generator.num_scalar_features)

    config.num_warehouses = reward_generator.num_warehouses

    allocation_model, optimizer, lr_scheduler = create_policy_model(config, reward_generator)
    value_model, optimizer_critic, lr_scheduler_critic = create_value_model(config, reward_generator)
    
    
    config.reward_generator = reward_generator
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    trainer = Trainer(allocation_model, value_model, optimizer, optimizer_critic, lr_scheduler, lr_scheduler_critic, loss_fn, reward_generator, config)
    # logger = Logger()

    for episode in tqdm(range(1, config.epoch_size)):
        
        try:
            trainer.train_episode(episode, training_batches, validation_batches)
        except Exception as e:
            print(f"Error in episode {episode}: {str(e)}")
            continue

if __name__ == "__main__":
      run(get_config())