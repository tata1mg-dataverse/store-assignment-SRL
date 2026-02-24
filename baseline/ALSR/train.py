import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

loss_fn_policy = nn.CrossEntropyLoss(reduction="none")

def actions(output, batch, is_explore=True):
    # output[batch['input'][:,:,-dat.num_warehouses:] <= 0.0] = -1e8
    probs = F.softmax(output,dim=-1)
    if(is_explore):
        actions = torch.cat([ torch.multinomial(probs[i, :int(count)], 1) for i, count in enumerate(batch['sku_count'])
        ])
    else:
        actions = np.concatenate([ torch.argmax(probs[i, :int(count)], dim=1).cpu() 
                            for i, count in enumerate(batch['sku_count'])
                            ]).tolist()
    return actions

def reward(actionsl, batch, reward_func):
    df = batch['df']
    if(isinstance(actionsl,torch.Tensor)):
        actionsl = np.concatenate(actionsl.cpu().tolist())
    df['actions'] = actionsl
    df['actions'] = df['actions'].astype(int)
    df = reward_func.get_reward(df)
    df['reward'] = df['reward'] * df.groupby('GROUP_ID').reward.transform(lambda x: not any(x==0))
    return df

def validate(model, dataset, opts, logger):
    model.eval()
    with torch.no_grad():
        for i, bat in enumerate(dataset):
            logits = model(bat['input'].to(opts.device),bat['mask'].to(opts.device))
            logits[bat['input'][:,:,-opts.num_warehouses:] <= 0.0] = -1e8
            actionsl = actions(logits, bat, is_explore=False)
            df = reward(actionsl, bat, opts.reward_generator)

            logger.log('validation_reward', df.reward.mean(), i,batch_type='validation')
            logger.log('avg_cost', df.cost.mean(), i,batch_type='validation')
            logger.log('avg_tat', df.tat.mean(), i,batch_type='validation')
            if(i==0):
                vals = (df.groupby('WAREHOUSE_CODE').GROUP_ID.nunique()/df.groupby('WAREHOUSE_CODE').GROUP_ID.nunique().sum())
                print(vals.sort_values(ascending=False)[0:3].to_dict())


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, reward_generator, logger, options):

    model.train()
    for i, batch in enumerate(tqdm(reward_generator.training_batches)):
        train_batch(model, optimizer, baseline, epoch, i, batch, options, logger )
    
    validate(model, reward_generator.validation_batches, options, logger)

    baseline.epoch_callback(model, epoch, logger)
    
    lr_scheduler.step()
    logger.step(epoch)

def train_batch(model, optimizer, baseline, epoch, step, batch, options, logger):
    train_logits = model(batch['input'],batch['mask'])
    train_logits[batch['input'][:,:,-options.num_warehouses:] <= 0.0] = -1e8
    baseline_logits = baseline.eval(batch['input'],batch['mask'])
    baseline_logits[batch['input'][:,:,-options.num_warehouses:] <= 0.0] = -1e8
    
    log_p = F.log_softmax(train_logits)

    train_actions = actions(train_logits, batch)
    train_df = reward(train_actions, batch, options.reward_generator)
    
    baseline_actions = actions(baseline_logits, batch)
    baseline_df = reward(baseline_actions, batch, options.reward_generator)

    actions_vals = torch.tensor(train_df.groupby(['GROUP_ID']).actions.apply(lambda x: np.append(np.array(x,dtype=np.int64),np.zeros(10-len(x),dtype=np.int64)))[train_df.GROUP_ID.unique()].to_list(),dtype=torch.int64).to(options.device)

    log_probs = F.log_softmax(train_logits, dim=-1)  #- loss_fn_policy(torch.permute(train_logits,(0,2,1)),actions_vals)
    log_likelihood = log_probs.gather( dim=2, index=actions_vals.unsqueeze(-1)).squeeze(-1)
    
    logp =  log_likelihood * batch['mask'] #log_likelihood.mean()#*(batch['mask']/(batch['mask'].sum(axis=1).reshape(-1,1)))
    policy_loss = logp.sum(dim=1)
    
    train_reward = torch.tensor(train_df.groupby('GROUP_ID').reward.mean().to_list()).to(device=options.device)
    baseline_reward = torch.tensor(baseline_df.groupby('GROUP_ID').reward.mean().to_list()).to(device=options.device)

    advantages = (train_reward - baseline_reward)
    loss = -(policy_loss * advantages ).mean()

    optimizer.zero_grad()
    loss.backward()
    
    nn.utils.clip_grad_norm_(model.parameters(), 1)

    logger.log('policy_loss', policy_loss.mean(), step)
    logger.log('loss', loss.mean(), step)
    logger.log('advantages', advantages.mean(), step)
    logger.log('training_reward', train_reward.mean(), step)
    logger.log('avg_cost', train_df.cost.mean(), step)
    logger.log('avg_tat', train_df.tat.mean(), step)

    optimizer.step()

    # Log values

def rollout(model, dataset, opts):
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            logits = model(bat['input'].to(opts.device),bat['mask'].to(opts.device))
            logits[bat['input'][:,:,-opts.num_warehouses:] <= 0.0] = -1e8
            actionsl = actions(logits, bat, is_explore=False)
            df = reward(actionsl, bat, opts.reward_generator)
            return torch.tensor(df.groupby('GROUP_ID').reward.mean().to_list()).to(opts.device)
    return torch.cat([
        eval_model_bat(bat)
        for bat
        in dataset
    ], 0)