import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import mlflow


class RLTrainer:
    """Reinforcement Learning Trainer for allocation model"""
    
    def __init__(self, allocation_model, value_model,
                 optimizer_policy_rl, optimizer_critic_rl, 
                 lr_scheduler_policy_rl, lr_scheduler_critic_rl,
                 loss_fn_policy,  dat ):
        
        self.allocation_model = allocation_model
        self.value_model = value_model
        self.optimizer_policy_rl = optimizer_policy_rl
        self.optimizer_critic_rl = optimizer_critic_rl
        self.lr_scheduler_policy_rl = lr_scheduler_policy_rl
        self.lr_scheduler_critic_rl = lr_scheduler_critic_rl
        self.loss_fn_policy = loss_fn_policy
        self.dat = dat
        self.device = dat.device
        self.max_seq_len = dat.max_seq_len
        
        # Tracking variables
        self.training_cost_dist = np.array([])
        self.training_tat_dist = np.array([])
        self.losses_logs = []
    
    def get_policy_logits(self, output, batch):
        """Extract policy logits based on SKU counts"""
        policy_logits = [
            output[i][:int(batch['sku_count'][i])].to('cpu') 
            for i in range(len(batch['sku_count']))
        ]
        return torch.cat(policy_logits)
    
    def apply_inventory_mask(self, output, batch):
        """Mask out warehouses with no inventory"""
        output[batch['input'][:, :, -self.dat.num_warehouses:] <= 0.0] = -1e8
        return output
    
    def sample_actions(self, probs, sku_counts):
        """Sample actions from probability distribution"""
        actions = torch.cat([
            torch.multinomial(probs[i, :int(count)], 1) 
            for i, count in enumerate(sku_counts)
        ])
        return actions.tolist()
    
    def prepare_actions_tensor(self, df):
        """Convert actions from dataframe to padded tensor"""
        print(df.columns)
        actions_vals = torch.tensor(
            df.groupby(['GROUP_ID']).actions.apply(
                lambda x: np.append(
                    np.array(x, dtype=np.int64),
                    np.zeros(self.max_seq_len - len(x), dtype=np.int64)
                )
            )[df.GROUP_ID.unique()].to_list(),
            dtype=torch.int64
        ).to(self.device)
        return actions_vals

    def compute_advantages(self, batch, probs, rewards):
        """Compute advantage values using value network"""
        action_values = torch.concat((batch['input'], probs), dim=-1).to(self.device)
        
        with torch.no_grad():
            self.value_model.train(mode=False)
            values = self.value_model(action_values, mask=batch['mask'])
        
        values = values.contiguous().reshape(-1)
        advantages = rewards - values.contiguous().reshape(-1)
        
        return advantages, values
    
    def update_policy_network(self, output, batch, actions_vals, advantages):
        """Perform policy gradient update"""
        log_probs = self.loss_fn_policy(
            torch.permute(output, (0, 2, 1)), 
            actions_vals
        )
        log_probs = log_probs * (batch['mask'] / (batch['mask'].sum(axis=1).reshape(-1, 1)))
        log_probs = log_probs.sum(axis=1)
        
        pi_loss = log_probs * advantages
        pi_loss = pi_loss.sum()
        
        self.optimizer_policy_rl.zero_grad()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.allocation_model.parameters(), 1)
        self.optimizer_policy_rl.step()
        
        return pi_loss, log_probs
    
    def update_value_network(self, batch, bnum):
        """Update value network (critic)"""
        with torch.no_grad():
            self.allocation_model.train(mode=False)
            output = self.allocation_model(batch['input'], mask=batch['mask'])
            
            policy_logits = self.get_policy_logits(output, batch)
            output = self.apply_inventory_mask(output, batch)
            probs = F.softmax(output, dim=-1)
        
        # Sample new actions
        actions = self.sample_actions(probs, batch['sku_count'])
        df = batch['df'].copy(deep=True)
        df['actions'] = np.concatenate(actions)
        
        
        # Compute rewards
        df = self.dat.get_reward(df)
        rewards = df.groupby(['GROUP_ID']).reward.mean()[df.GROUP_ID.unique()]
        rewards = torch.tensor(rewards.to_list(), dtype=torch.float).to(self.device)
        
        # Train value network
        action_values = torch.concat((batch['input'], probs), dim=-1).to(self.device)
        self.value_model.train(mode=True)
        values = self.value_model(action_values, mask=batch['mask'])
        values = values.contiguous().reshape(-1)
        
        vf_loss = F.mse_loss(values, rewards, reduction="none").mean()
        
        self.optimizer_critic_rl.zero_grad()
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1)
        self.optimizer_critic_rl.step()
        
        return vf_loss
    
    def train_batch(self, batch, bnum, episodes):
        """Train on a single batch"""
        # Forward pass through policy network
        output = self.allocation_model(batch['input'], mask=batch['mask'])
        policy_logits = self.get_policy_logits(output, batch)
        output = self.apply_inventory_mask(output, batch)
        probs = F.softmax(output, dim=-1)
        
        # Sample actions
        actions = self.sample_actions(probs, batch['sku_count'])
        df = batch['df'].copy(deep=True)
        df['actions'] = np.concatenate(actions)
        
        
        # Compute rewards
        df = self.dat.get_reward(df)
        rewards = df.groupby(['GROUP_ID']).reward.mean()[df.GROUP_ID.unique()]
        rewards = torch.tensor(rewards.to_list(), dtype=torch.float).to(self.device)

        # Compute advantages
        advantages, values = self.compute_advantages(batch, probs, rewards)
        
        # Prepare actions tensor
        actions_vals = self.prepare_actions_tensor(df)
        
        # Update policy network
        pi_loss, log_probs = self.update_policy_network(output, batch, actions_vals, advantages)
        
        # Update value network (every episode)
        vf_loss = torch.tensor(0., dtype=torch.float)
        if episodes % 1 == 0:
            vf_loss = self.update_value_network(batch, bnum)
        
        # Compute metrics
        cost_ = df.groupby(['GROUP_ID']).cost_level.sum().mean()
        tat_ = df.groupby(['GROUP_ID']).tat.mean().mean()
        
        # Get distribution statistics
        cost_dist = df.groupby(['GROUP_ID']).cost_level.sum().values
        tat_dist = df.groupby(['GROUP_ID']).tat.mean().values
        
        # Create log entry
        log_entry = [
            episodes,
            self.optimizer_policy_rl.param_groups[0]['lr'],
            self.optimizer_critic_rl.param_groups[0]['lr'],
            pi_loss.detach().cpu().numpy(),
            log_probs.mean().detach().cpu().numpy(),
            advantages.mean().detach().cpu().numpy(),
            (log_probs * advantages).mean().detach().cpu().numpy(),
            vf_loss.detach().cpu().numpy(),
            values.mean().detach().cpu().numpy(),
            cost_,
            tat_,
            torch.mean(rewards).detach().cpu().numpy()
        ]
        
        batch_info = {
            'df': df,
            'pi_loss': pi_loss,
            'vf_loss': vf_loss,
            'cost_dist': cost_dist,
            'tat_dist': tat_dist,
            'log_entry': log_entry
        }
        
        return batch_info, False
    
    def validate_batch(self, validation_batch):
        """Validate on a single batch"""
        valid_logits = self.allocation_model(validation_batch['input'],validation_batch['mask'])
        valid_logits = self.apply_inventory_mask(valid_logits, validation_batch)
        probs = F.softmax(valid_logits, dim=-1)

        sku_count = validation_batch['sku_count']

        actions_vals = probs.argmax(dim=-1)
        actions = [actions_vals[y][0:int(sku_count[y])] for y in range(len(sku_count))]
        actions = torch.cat(actions)
        
        df = validation_batch['df']
        df['actions'] = actions.cpu().numpy()
        df = self.dat.get_reward(df)

        log_probs = self.loss_fn_policy(
            torch.permute(valid_logits, (0, 2, 1)), 
            actions_vals
        )
        log_probs = log_probs * (validation_batch['mask'] / (validation_batch['mask'].sum(axis=1).reshape(-1, 1)))
        log_probs = log_probs.sum(axis=1)
        pi_loss = log_probs.sum()
        
        # Aggregate metrics
        bat_df_agg = df.groupby(['GROUP_ID']).agg({
            'cost_level': 'sum',
            'tat': 'mean',
            'reward': 'mean',
            'sku_count': 'mean'
        }).mean()
        
        val_log_entry = [
            float(pi_loss.detach().cpu()),
            bat_df_agg.cost_level,
            bat_df_agg.tat,
            bat_df_agg.reward
        ]
        
        
        return val_log_entry, pi_loss
    
    def train_episode(self, episodes, training_batches, validation_batches):
        """Train for one episode"""
        ep_loss = {
            "training": {"policy_net": [], "value_net": []},
            "validation": {"policy_net": [], "value_net": []}
        }
        val_log = []
        logs = []
        all_df = pd.DataFrame()
        
        training_cost_dist_batch = np.array([])
        training_tat_dist_batch = np.array([])
        break_status = False
        
        # Training phase
        for bnum, batch in enumerate(training_batches):
            batch_info, break_flag = self.train_batch(batch, bnum, episodes)
            
            if break_flag:
                break_status = True
                break
            
            # Accumulate results
            ep_loss['training']['value_net'].append(float(batch_info['vf_loss'].detach().cpu()))
            ep_loss['training']['policy_net'].append(float(batch_info['pi_loss'].detach().cpu()))
            
            training_cost_dist_batch = np.concatenate([training_cost_dist_batch, batch_info['cost_dist']])
            training_tat_dist_batch = np.concatenate([training_tat_dist_batch, batch_info['tat_dist']])
            
            all_df = pd.concat([all_df, batch_info['df']])
            logs.append(batch_info['log_entry'])
            
            # Print initial batch statistics
            if bnum == 0 and episodes == 0:
                vals = batch_info['df'].WAREHOUSE_CODE.value_counts(normalize=True) * 100
                print(vals.to_dict(), batch_info['df'].groupby(['GROUP_ID']).agg({
                    'cost_level': 'sum',
                    'tat': 'mean'
                }).mean())
        
        if break_status:
            return None, True
        
        # Update learning rate schedulers
        self.lr_scheduler_critic_rl.step()
        self.lr_scheduler_policy_rl.step()
        
        # Compute and save distribution statistics
        quantiles = [item / 10 for item in range(1, 10)]
        dist_cost = np.quantile(training_cost_dist_batch, quantiles).round(1)
        dist_tat = np.quantile(training_tat_dist_batch, quantiles).round(1)
        
        self.training_cost_dist = np.concatenate([self.training_cost_dist, dist_cost])
        self.training_tat_dist = np.concatenate([self.training_tat_dist, dist_tat])
        
        np.save('training_cost_dist.npy', self.training_cost_dist)
        np.save('training_tat_dist.npy', self.training_tat_dist)
        
        # Print training statistics
        if episodes % 10 == 1:
            vals = all_df.WAREHOUSE_CODE.value_counts(normalize=True) * 100
            uactions = len(all_df.actions.unique())
            print(f"\nEpisode {episodes}, \nCost Dist: {dist_cost}\nTAT Dist: {dist_tat}")
            print(f"episode-{episodes}, LR_P {self.lr_scheduler_policy_rl.get_last_lr()}, "
                  f"LR_C {self.lr_scheduler_critic_rl.get_last_lr()}, "
                  f"policy_loss {sum(ep_loss['training']['policy_net'])}, "
                  f"uniq: {uactions}, train: {vals[0:3].to_dict()}", end=" | ")
            print(episodes, all_df.groupby([all_df.ORDER_DATE.dt.date, 'GROUP_ID']).agg({
                'cost_level': 'sum',
                'tat': 'mean'
            }).groupby(['ORDER_DATE']).mean())
        
        # Validation phase
        self.allocation_model.eval()
        topk = []
        
        with torch.no_grad():
            for bnum_valid, validation_batch in enumerate(validation_batches):
                val_log_entry, pi_loss = self.validate_batch(
                    validation_batch, 
                )
                
                ep_loss['validation']['policy_net'].append(float(pi_loss.detach().cpu()))
                val_log.append(val_log_entry)
        
        # Print validation statistics
        if episodes % 10 == 0:
            print(f"validation_policy_loss {sum(ep_loss['validation']['policy_net'])}, "
                  f"vals: {vals[0:3].to_dict()} | topk_percentage {np.mean(topk)}")
        
        # Log metrics to MLflow
        metrics = ['episode', 'lr_policy', 'lr_critic', 'policy_loss', 'log_prob_mean', 
                   'advantages_mean', 'LA_mean', 'value_loss', 'value_mean', 
                   'training_cost', 'training_tat', 'training_reward']
        metrics += ['validation_loss', 'validation_cost', 'validation_tat', 
                    'validation_reward']
        
        combined_log = np.concatenate([np.mean(logs, axis=0), np.mean(val_log, axis=0)])
        self.losses_logs.append(combined_log)
        
        for index, met in enumerate(metrics):
            mlflow.log_metric(met, self.losses_logs[-1][index], step=episodes)
        
        # Save models at specified episodes
        fsave = json.load(open('file_save.json', 'r'))["rl"]
        if episodes in fsave:
            mlflow.pytorch.log_model(self.allocation_model, f'allocation_model_{episodes}')
            mlflow.pytorch.log_model(self.value_model, f'value_model_{episodes}')
        
        self.allocation_model.train(mode=True)
        
        return ep_loss, False
