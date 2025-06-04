import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Neural Network Training Script")

    parser.add_argument('-wp', '--wandb_project', default='myprojectname',
                        help='Project name')
    parser.add_argument('-we', '--wandb_entity', default='myname',
                        help='wandb entity')

    parser.add_argument('-d', '--dataset', default='fashion_mnist', choices=['mnist', 'fashion_mnist'],
                        help='dataset to train on')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=4,
                        help='batch size')
    
    parser.add_argument('-l', '--loss', default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'],
                        help='loss function')
    parser.add_argument('-o', '--optimizer', default='sgd',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        help='Optimizer')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1)
    
    parser.add_argument('-m', '--momentum', type=float, default=0.5)
    parser.add_argument('-beta', '--beta', type=float, default=0.5)
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5)
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5)
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-6)

    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-w_i', '--weight_init', default='random', choices=['random', 'Xavier'])
    
    parser.add_argument('-nhl', '--num_layers', type=int, default=1)
    parser.add_argument('-sz', '--hidden_size', type=int, default=4)
    parser.add_argument('-a', '--activation', default='sigmoid',
                        choices=['identity', 'sigmoid', 'tanh', 'ReLU'])

    return parser.parse_args()
