import argparse


def get_config(args=None):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Allocation Problem with Reinforcement Learning")
    parser.add_argument('--epoch_size', type=int, default=500, help='Number of instances per epoch during training')
    parser.add_argument('--learning_rate', type=float, default=0.04, help='Learning rate for the optimizer')
    parser.add_argument('--kl_beta', type=float, default=5, help='KL divergence coefficient')
    parser.add_argument('--alpha', type=float, default=0.25, help='Alpha coefficient')
    parser.add_argument('--fast_delivery_alpha', type=float, default=0.75, help='fast_delivery alpha coefficient')
    parser.add_argument('--bl_alpha', type=float, default=0.05, help='BL alpha coefficient')
    parser.add_argument('--policy_model_location', type=str, default='', help='Policy model location')
    parser.add_argument('--value_model_location', type=str, default='', help='Value model location')

    config = parser.parse_args(args)
    config.device = 'cuda'
    config.path = 'data'
    config.max_seq_len = 10
    config.num_of_train_batch = 40
    config.num_of_validation_batch = 5

    return config