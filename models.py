from model import AllocationModel
from model import ValueNetwork
from torch.optim.lr_scheduler import SequentialLR
import torch.optim as optim
import torch

def create_policy_model(config, reward_generator, path=None):
    num_scalar_features = reward_generator.num_scalar_features
    mlp_output_dim = reward_generator.num_warehouses
    mlp_hidden_layers = [512, 256, 128, 128, 64]
    transformer_output_dim = 512
    num_attention_heads = 8
    transformer_ff_dim = 128
    allocation_output_dim = reward_generator.num_warehouses
    dropout = 0.2
    num_encoder_layers = 8
    if path:
        allocation_model = torch.load(path, weights_only=False)

        optimizer = optim.Adam(allocation_model.parameters(), lr=1e-5)
        lr_scheduler_policy_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-6, T_max=100)
        lr_scheduler_policy_warm_up = optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=2)
        lr_scheduler = SequentialLR(
            optimizer, schedulers=[lr_scheduler_policy_warm_up, lr_scheduler_policy_cosine],
            milestones = [2]
        )

        num_params = sum(p.numel() for p in allocation_model.parameters() if p.requires_grad)
        print(f"Number of parameters : {num_params}")
        allocation_model.train(mode=True)
        
    else:
        allocation_model = AllocationModel(
        num_scalar_features, mlp_output_dim, mlp_hidden_layers,
        transformer_output_dim, num_attention_heads, transformer_ff_dim,
        allocation_output_dim, dropout, num_encoder_layers=num_encoder_layers).to(config.device)

        num_params = sum(p.numel() for p in allocation_model.parameters() if p.requires_grad)
        print(f"Number of parameters : {num_params}")

        optimizer = optim.Adam(allocation_model.parameters(), lr=config.learning_rate)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-7, T_max=200)
        
        allocation_model.train(mode=True)
    return allocation_model, optimizer, lr_scheduler

def create_value_model(config, reward_generator, path = None):
    input_dim = reward_generator.num_scalar_features + config.num_warehouses
    mlp_hidden_layers = [512, 256, 128, 128, 64]
    
    max_seq_len = config.max_seq_len
    d_model = 512
    num_heads = 16
    d_ff = 128
    dropout = 0.3
    num_encoder_layers = 8
    if path:
        value_model = torch.load(path, weights_only=False)

        num_params = sum(p.numel() for p in value_model.parameters() if p.requires_grad)
        print(f"Number of parameters : {num_params}")

        optimizer_critic = optim.Adam(value_model.parameters(), lr= config.learning_rate)
        lr_scheduler_critic_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer_critic, eta_min=1e-6, T_max=100)
        lr_scheduler_critic_warm_up = optim.lr_scheduler.ConstantLR(optimizer_critic, factor=0.1, total_iters=2)
        lr_scheduler_critic = SequentialLR(
            optimizer_critic, schedulers=[lr_scheduler_critic_warm_up, lr_scheduler_critic_cosine],
            milestones = [2]
        )

        value_model = value_model.train(mode=True)

    else:
        value_model = ValueNetwork(input_dim, mlp_hidden_layers, max_seq_len, d_model, num_heads, d_ff, dropout, num_encoder_layers).to(config.device)
        optimizer_critic = optim.Adam(value_model.parameters(), lr= 1e-03)
        lr_scheduler_critic = optim.lr_scheduler.CosineAnnealingLR(optimizer_critic, eta_min=1e-7, T_max=200)

        num_params = sum(p.numel() for p in value_model.parameters() if p.requires_grad)
        print(f"Number of parameters : {num_params}")

        

        value_model = value_model.train(mode=True)
    return value_model, optimizer_critic, lr_scheduler_critic