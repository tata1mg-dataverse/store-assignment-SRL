import numpy as np
import pandas as pd 
import mlflow
import torch
import json 

from .reinforce_baseline import RolloutBaseline
from input_generator import DataPreparator
from reward_generator import RewardCalculator
from models import create_policy_model
from .train  import train_epoch
from .utils import Logger
from config import get_config


def run(config):
    path = config.path
    orders = pd.read_parquet(f'{path}/orders.parquet')
    order_skus = pd.read_parquet(f'{path}/order_skus.parquet')
    sku_vol = pd.read_parquet(f'{path}/sku_vol.parquet')

    df = order_skus.merge(orders,on=['GROUP_ID','ORDER_ID'])
    df = df.merge(sku_vol, on='SKU_ID',how='left')
    df['ORDER_DATE'] = pd.to_datetime(df['ORDER_DATE'],format='mixed')
    df['IS_fast_delivery'] = df['DELIVERY_TYPE'].map({'fast_delivery':1,'slow_delivery':0})

    df['COLD_STORAGE'] = [np.random.random() < 0.05 for x in range(len(df))]
    
    df['real_actions'] = df['WAREHOUSE_CODE']

    preparator = DataPreparator(path)
    df = df[df.CENTRAL_PINCODE!=-1]
    df = preparator.prepare_data(df)
    df = df[df.inputs.map(len)==149]
    
    training_batches = preparator.create_batches(df[df.ORDER_DATE<'2024-05-10'],149, num_of_batch=80 ,device = config.device)
    validation_batches = preparator.create_batches(df[df.ORDER_DATE>='2024-05-10'],149, num_of_batch=20, device = config.device)


    class_weights = df.shape[0] / (df.WAREHOUSE_CODE.nunique() * np.bincount(df['real_actions']))

    reward_generator = RewardCalculator(path, config.alpha, config.fast_delivery_alpha)
    
    reward_generator.validation_batches = validation_batches
    reward_generator.training_batches = training_batches
    reward_generator.class_weights = class_weights

    reward_generator.class_weights[reward_generator.class_weights==np.inf] = 0
    reward_generator.class_weights = torch.tensor(reward_generator.class_weights).to(config.device)

    config.reward_generator = reward_generator

    reward_generator.num_scalar_features = training_batches[0]['input'].shape[2]
    print(reward_generator.num_scalar_features)

    config.num_warehouses = reward_generator.num_warehouses

    allocation_model, optimizer, lr_scheduler = create_policy_model(config, reward_generator)
    baseline = RolloutBaseline(allocation_model, reward_generator, config)
    config.reward_generator = reward_generator

    mlflow.set_tracking_uri('http://localhost:2000/')
    mlflow.set_experiment("baseline_ALSR")
    mlflow.autolog()
    with mlflow.start_run():
        logger = Logger(mlflow)
        for epoch in range(0, config.epoch_size):
                train_epoch(
                    allocation_model,
                    optimizer,
                    baseline,
                    lr_scheduler,
                    epoch,
                    reward_generator,
                    logger,
                    config
                )
                fsave = json.load(open('file_save.json','r'))["rl"]
                if(epoch in fsave ):
                    mlflow.pytorch.log_model(allocation_model,f'allocation_model_{epoch}')


if __name__ == "__main__":
      run(get_config())