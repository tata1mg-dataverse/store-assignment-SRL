import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import mlflow
import json


class Trainer:
    """Handles the training and validation of allocation and value models."""
    
    def __init__(
        self,
        allocation_model,
        value_model,
        optimizer_policy,
        optimizer_critic,
        lr_scheduler_policy,
        lr_scheduler_critic,
        loss_fn_policy,
        dat,
        config,
    ):
        self.allocation_model = allocation_model
        self.value_model = value_model
        self.optimizer_policy = optimizer_policy
        self.optimizer_critic = optimizer_critic
        self.lr_scheduler_policy = lr_scheduler_policy
        self.lr_scheduler_critic = lr_scheduler_critic
        self.loss_fn_policy = loss_fn_policy
        self.dat = dat
        self.device = config.device
        self.max_seq_len = config.max_seq_len
        self.config = config

    def initialize_episode_losses(self):
        """Initialize loss tracking structure for an episode."""
        return {
            "training": {"policy_net": [], "value_net": []},
            "validation": {"policy_net": [], "value_net": []}
        }
    
    def get_allocation_output(self, batch):
        """Get allocation model output with masked invalid actions(actions with zero inventory)."""
        with torch.no_grad():
            self.allocation_model.train(mode=False)
            logits = self.allocation_model(batch['input'].to(self.device), batch['mask'].to(self.device))
        
        # Mask unavailable warehouses
        logits[batch['input'][:, :, -self.dat.num_warehouses:] == 0.0] = -1e8
        probs = F.softmax(logits, dim=-1)
        
        return logits, probs
    
    def sample_actions(self, probs, batch):
        """Sample actions from probability distribution."""
        actions = torch.cat([
            torch.multinomial(probs[i, :int(count)], 1) 
            for i, count in enumerate(batch['sku_count'])
        ]).tolist()
        
        return actions
    def create_filters(self, batch):
        """Create filters for invalid warehouse selections."""
        filters = batch['input'][:, :, -self.dat.num_warehouses:] > 0.0
        # filter2 = batch['input'][:, :, 6:6+self.dat.num_warehouses] > 0
        
        # filters = filter1 | filter2
        # filters = ~filters
        
        return filters
    
    def compute_rewards(self, batch, actions):
        """Compute rewards based on sampled actions."""
        df = batch['df']
        df['actions'] = np.concatenate(actions)
        df = self.dat.get_reward(df)
        
        rewards = df.groupby(['GROUP_ID']).reward.mean()
        return df, rewards
    
    def prepare_action_tensors(self, df, batch):
        """Prepare action tensors for loss calculation."""
        actions_vals = torch.tensor(
            df.groupby(['GROUP_ID']).real_actions.apply(
                lambda x: np.append(
                    np.array(x, dtype=np.int64),
                    np.zeros(self.max_seq_len - len(x), dtype=np.int64)
                )
            )[df.GROUP_ID.unique()].to_list(),
            dtype=torch.int64
        ).to(self.device)
        
        return actions_vals
    
    def train_value_network(self, batch, probs, rewards):
        """Train the value network."""
        action_values = torch.concat(
            (batch['input'], probs), dim=-1
        ).to(self.device)
        
        values = self.value_model(action_values, mask=batch['mask'])
        values = values.contiguous().reshape(-1)
        
        self.optimizer_critic.zero_grad()
        vf_loss = F.mse_loss(values, rewards, reduction="none")
        vf_loss.mean().backward()
        self.optimizer_critic.step()
        
        return float(vf_loss.mean().detach().cpu())
    
    def train_policy_network(self, batch, df, filters):
        """Train the policy network."""
        self.optimizer_policy.zero_grad()
        self.allocation_model.train(mode=True)
        
        logits = self.allocation_model(batch['input'], mask=batch['mask'])
        actions_vals = self.prepare_action_tensors(df, batch)
        
        logits[filters] = -1e8
        
        log_probs = self.loss_fn_policy(
            torch.permute(logits, (0, 2, 1)), 
            actions_vals
        )
        
        log_probs = log_probs * (
            batch['mask'] / batch['mask'].sum(axis=1).reshape(-1, 1)
        )
        log_probs = log_probs.sum(axis=1)
        
        pi_loss = log_probs
        pi_loss.mean().backward()
        self.optimizer_policy.step()
        
        probs = F.softmax(logits, dim=-1)
        
        return float(pi_loss.mean().detach().cpu()), probs
    
    def train_batch(self, batch, bnum):
        """Train on a single batch."""
        # Get allocation output and sample actions
        logits, probs = self.get_allocation_output(batch)
        actions = self.sample_actions(probs, batch)
        
        # Compute rewards
        df, rewards_series = self.compute_rewards(batch, actions)
        
        # Convert rewards to tensor
        actions_tensor = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(
            rewards_series.to_list(), dtype=torch.float
        ).to(self.device)
        
        # Train value network
        vf_loss = self.train_value_network(batch, probs, rewards)
        
        # Train policy network
        filters = self.create_filters(batch)
        pi_loss, probs = self.train_policy_network(batch, df, filters)
        
        # Calculate metrics
        # topk = self.dat.is_topk_vendors(probs, df)
        
        return pi_loss, vf_loss, df
    
    def validate_batch(self, validation_batch):
        """Validate on a single batch."""
        valid_logits, probs = self.get_allocation_output(validation_batch)

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
        log_probs = log_probs * (
            validation_batch['mask'] / 
            validation_batch['mask'].sum(axis=1).reshape(-1, 1)
        )
        log_probs = log_probs.sum(axis=1)
        
        pi_loss_v = log_probs
        probs = F.softmax(valid_logits, dim=-1)
        # topk = self.dat.is_topk_vendors(probs, bat_df)
        
        aggregated = df.groupby(['GROUP_ID']).agg({
            'cost_level': 'sum',
            'tat': 'mean',
            'reward': 'mean',
            'sku_count': 'mean'
        }).mean()
        
        return {
            'policy_loss': float(pi_loss_v.mean().detach().cpu()),
            'cost_level': aggregated.cost_level,
            'tat': aggregated.tat,
            'reward': aggregated.reward,
            # 'topk': topk.cpu().numpy(),
        }
    
    def log_metrics(self, episode, train_logs, val_logs):
        """Log metrics to MLflow."""
        train_metrics = np.mean(train_logs, axis=0)
        val_metrics = np.mean(val_logs, axis=0)
        
        combined_metrics = np.concatenate([train_metrics, val_metrics])
        print(combined_metrics)
        
        metric_names = [
            'episode', 'policy_loss', 'value_loss', 'value_mean',
             'validation_loss', 'cost_val', 
            'tat_val', 'reward_val'
        ]
        
        for idx, name in enumerate(metric_names):
            mlflow.log_metric(name, combined_metrics[idx], step=episode)
        
        mlflow.log_metric(
            'policy_lr', 
            self.optimizer_policy.param_groups[0]['lr'], 
            step=episode
        )
        mlflow.log_metric(
            'critic_lr', 
            self.optimizer_critic.param_groups[0]['lr'], 
            step=episode
        )
    
    def save_models(self, episode):
        """Save models at specified episodes."""
        with open('file_save.json', 'r') as f:
            fsave = json.load(f)["supervised"]
        
        if episode in fsave:
            mlflow.pytorch.log_model(
                self.allocation_model, 
                f'allocation_model_{episode}'
            )
            mlflow.pytorch.log_model(
                self.value_model, 
                f'value_model_{episode}'
            )
    
    def train_episode(self, episode, training_batches, validation_batches):
        """Train a single episode."""
        ep_loss = self.initialize_episode_losses()
        val_logs = []
        train_logs = []
        
        # Training loop
        for bnum, batch in enumerate(training_batches):
            pi_loss, vf_loss, df = self.train_batch(batch, bnum)
            
            ep_loss['training']['policy_net'].append(pi_loss)
            ep_loss['training']['value_net'].append(vf_loss)
            
            train_logs.append([
                episode, pi_loss, vf_loss, 
                0.0,  # placeholder for value_mean
                # topk.cpu().numpy()
            ])
        
        # Print training summary
        print(
            f"episode-{episode}, "
            f"LR_P-{self.lr_scheduler_policy.get_last_lr()}, "
            f"LR_C-{self.lr_scheduler_critic.get_last_lr()}, "
            f"policy_loss {sum(ep_loss['training']['policy_net'])}, "
            f"value_loss {np.mean(ep_loss['training']['value_net'])}",
            end=" | "
        )
        
        # Update learning rates
        self.lr_scheduler_critic.step()
        self.lr_scheduler_policy.step()
        
        # Validation loop
        self.allocation_model.eval()
        # topk_vals = []
        
        
        with torch.no_grad():
            for bnum_valid, validation_batch in enumerate(validation_batches):
                val_result = self.validate_batch(validation_batch)
                
                ep_loss['validation']['policy_net'].append(
                    val_result['policy_loss']
                )
                # topk_vals.append(val_result['topk'])
                
                
                val_logs.append([
                    val_result['policy_loss'],
                    val_result['cost_level'],
                    val_result['tat'],
                    val_result['reward'],
                    # np.mean(topk_vals)
                ])
        
        # Log metrics
        losses_logs = [np.concatenate([
            np.mean(train_logs, axis=0), 
            np.mean(val_logs, axis=0)
        ])]
        self.log_metrics(episode, train_logs, val_logs)
        
        # Print validation summary
        print(
            f"validation_policy_loss {sum(ep_loss['validation']['policy_net'])} "
            # f"| topk_percentage {np.mean(topk_vals)}"
        )
        
        # Save models if needed
        self.save_models(episode)
        
        # Set model back to training mode
        self.allocation_model.train(mode=True)
        
        return ep_loss, losses_logs