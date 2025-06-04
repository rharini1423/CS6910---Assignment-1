from args import get_args  
from train import train, fwd
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import wandb
import numpy as np

def main():
    args = get_args()

    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    wandb.config.update(vars(args))  

    if args.dataset == 'mnist':
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    else:
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    X_train_full = X_train_full.reshape(-1, 28*28) / 255.0
    X_test = X_test.reshape(-1, 28*28) / 255.0

    encoder = OneHotEncoder(sparse_output=False)
    y_train_full_enc = encoder.fit_transform(y_train_full.reshape(-1, 1))
    y_test_enc = encoder.transform(y_test.reshape(-1, 1))

    
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full_enc, test_size=0.1, random_state=42)

   
    hidden_layers = [args.hidden_size] * args.num_layers
    layer_sizes = [784] + hidden_layers + [10]

    model = train(
        X_train, y_train,
        X_val, y_val,
        sizes=layer_sizes,
        opt_name=args.optimizer,
        epochs=args.epochs,
        bs=args.batch_size,
        lr=args.learning_rate,
        wd=args.weight_decay,
        init=args.weight_init,
        act=args.activation,
        loss_fn=args.loss
    )


    _, test_act = fwd(X_test, model[0], model[1], args.activation)
    test_preds = np.argmax(test_act[-1], axis=1)
    test_labels = np.argmax(y_test_enc, axis=1)
    test_acc = np.mean(test_preds == test_labels)

    wandb.log({'test_accuracy': test_acc})
    print(f'Test Accuracy: {test_acc:.2f}%')
    wandb.finish()

if __name__ == '__main__':
    main()
