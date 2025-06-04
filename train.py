import numpy as np
import wandb

def relu(x):
    return np.maximum(0,x)

def relu_d(x):
    return (x>0).astype(float)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_d(x):
    s=sigmoid(x)
    return s*(1-s)

def tanh(x):
    return np.tanh(x)

def tanh_d(x):
    return 1-np.tanh(x)**2

def softmax(x):
    e=np.exp(x-np.max(x,axis=1,keepdims=True))
    return e/np.sum(e,axis=1,keepdims=True)

def loss(y_hat, y):
    m = y.shape[0]
    if loss_type == 'squared':
        return 0.5 * np.sum((y_hat - y) ** 2) / m
    else:
        return -np.sum(y * np.log(y_hat + 1e-8)) / m

def init_params(sz,mtd):
    w,b=[],[]
    for i in range(len(sz)-1):
        if mtd=='xavier':
            lim=np.sqrt(6/(sz[i]+sz[i+1]))
            wi=np.random.uniform(-lim,lim,(sz[i],sz[i+1]))
        else:
            wi=np.random.randn(sz[i],sz[i+1])*np.sqrt(2./sz[i])
        bi=np.zeros((1,sz[i+1]))
        w.append(wi)
        b.append(bi)
    return w,b

def fwd(x,w,b,act):
    z,a=[],[x]
    cur=x
    for i in range(len(w)-1):
        lin=np.dot(cur,w[i])+b[i]
        z.append(lin)
        if act=='sigmoid':
            cur=sigmoid(lin)
        elif act=='tanh':
            cur=tanh(lin)
        else:
            cur=relu(lin)
        a.append(cur)
    out=np.dot(cur,w[-1])+b[-1]
    z.append(out)
    a.append(softmax(out))
    return z,a

def bwd(y,w,z,a,act):
    gw=[None]*len(w)
    gb=[None]*len(w)
    m=y.shape[0]
    d=a[-1]-y
    gw[-1]=np.dot(a[-2].T,d)/m
    gb[-1]=np.sum(d,axis=0,keepdims=True)/m
    for i in reversed(range(len(w)-1)):
        if act=='sigmoid':
            d=np.dot(d,w[i+1].T)*sigmoid_d(z[i])
        elif act=='tanh':
            d=np.dot(d,w[i+1].T)*tanh_d(z[i])
        else:
            d=np.dot(d,w[i+1].T)*relu_d(z[i])
        gw[i]=np.dot(a[i].T,d)/m
        gb[i]=np.sum(d,axis=0,keepdims=True)/m
    return gw,gb

def init_opt(w):
    s={}
    for i in range(len(w)):
        shp=w[i].shape
        s[i]={
            'vw':np.zeros_like(w[i]),
            'vb':np.zeros((1,shp[1])),
            'sw':np.zeros_like(w[i]),
            'sb':np.zeros((1,shp[1]))
        }
    return s

def sgd(w,b,gw,gb,opt):
    lr=opt['lr']
    for i in range(len(w)):
        w[i]-=lr*gw[i]
        b[i]-=lr*gb[i]
    return w,b

def mom(w,b,gw,gb,opt):
    lr=opt['lr']
    b1=opt['beta1']
    for i in range(len(w)):
        s=opt['state'][i]
        s['vw']=b1*s['vw']+(1-b1)*gw[i]
        s['vb']=b1*s['vb']+(1-b1)*gb[i]
        w[i]-=lr*s['vw']
        b[i]-=lr*s['vb']
    return w,b

def nest(w,b,gw,gb,opt):
    lr=opt['lr']
    b1=opt['beta1']
    for i in range(len(w)):
        s=opt['state'][i]
        pvw=s['vw']
        pvb=s['vb']
        s['vw']=b1*pvw+(1-b1)*gw[i]
        s['vb']=b1*pvb+(1-b1)*gb[i]
        w[i]-=lr*(b1*pvw+(1-b1)*gw[i])
        b[i]-=lr*(b1*pvb+(1-b1)*gb[i])
    return w,b

def rms(w,b,gw,gb,opt):
    lr=opt['lr']
    b2=opt['beta2']
    eps=opt['eps']
    for i in range(len(w)):
        s=opt['state'][i]
        s['sw']=b2*s['sw']+(1-b2)*(gw[i]**2)
        s['sb']=b2*s['sb']+(1-b2)*(gb[i]**2)
        w[i]-=lr*gw[i]/(np.sqrt(s['sw'])+eps)
        b[i]-=lr*gb[i]/(np.sqrt(s['sb'])+eps)
    return w,b

def adam(w,b,gw,gb,opt):
    lr=opt['lr']
    b1=opt['beta1']
    b2=opt['beta2']
    eps=opt['eps']
    t=opt['t']
    for i in range(len(w)):
        s=opt['state'][i]
        s['vw']=b1*s['vw']+(1-b1)*gw[i]
        s['sw']=b2*s['sw']+(1-b2)*(gw[i]**2)
        s['vb']=b1*s['vb']+(1-b1)*gb[i]
        s['sb']=b2*s['sb']+(1-b2)*(gb[i]**2)
        vcw=s['vw']/(1-b1**t)
        scw=s['sw']/(1-b2**t)
        vcb=s['vb']/(1-b1**t)
        scb=s['sb']/(1-b2**t)
        w[i]-=lr*vcw/(np.sqrt(scw)+eps)
        b[i]-=lr*vcb/(np.sqrt(scb)+eps)
    opt['t']+=1
    return w,b

def nadam(w,b,gw,gb,opt):
    lr=opt['lr']
    b1=opt['beta1']
    b2=opt['beta2']
    eps=opt['eps']
    t=opt['t']
    for i in range(len(w)):
        s=opt['state'][i]
        s['vw']=b1*s['vw']+(1-b1)*gw[i]
        s['sw']=b2*s['sw']+(1-b2)*(gw[i]**2)
        s['vb']=b1*s['vb']+(1-b1)*gb[i]
        s['sb']=b2*s['sb']+(1-b2)*(gb[i]**2)
        vcw=(b1*s['vw']+(1-b1)*gw[i])/(1-b1**t)
        scw=s['sw']/(1-b2**t)
        vcb=(b1*s['vb']+(1-b1)*gb[i])/(1-b1**t)
        scb=s['sb']/(1-b2**t)
        w[i]-=lr*vcw/(np.sqrt(scw)+eps)
        b[i]-=lr*vcb/(np.sqrt(scb)+eps)
    opt['t']+=1
    return w,b

optimizers = {
    'sgd': sgd,
    'momentum': mom,
    'nesterov': nest,
    'rmsprop': rms,
    'adam': adam,
    'nadam': nadam
}


def train(Xtr, ytr, Xval, yval, sizes, opt_name='adam', epochs=10, bs=64, lr=0.001, wd=0.0, init='random', act='relu', loss_fn='cross_entropy'):
    global loss_type
    loss_type = loss_fn
    w, b = init_params(sizes, init)
    opt = {'name': opt_name, 'lr': lr, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8, 't': 1, 'state': init_opt(w) if opt_name != 'sgd' else None}

    n = Xtr.shape[0]
    for e in range(epochs):
        idx = np.random.permutation(n)
        Xs, ys = Xtr[idx], ytr[idx]
        for i in range(0, n, bs):
            xb, yb = Xs[i:i+bs], ys[i:i+bs]
            pre, actv = fwd(xb, w, b, act)
            gw, gb = bwd(yb, w, pre, actv, act)
            if wd > 0:
                for j in range(len(gw)):
                    gw[j] += wd * w[j]
            if opt_name == 'sgd':
                w, b = sgd(w, b, gw, gb, opt)
            else:
                w, b = optimizers[opt_name](w, b, gw, gb, opt)
        _, train_act = fwd(Xtr, w, b, act)
        _, val_act = fwd(Xval, w, b, act)
        loss_val = loss(train_act[-1], ytr)
        acc = (np.argmax(val_act[-1], axis=1) == np.argmax(yval, axis=1)).mean() * 100
        wandb.log({"epoch": e+1, "loss": loss_val, "val_accuracy": acc})
        print(f"Epoch {e+1}/{epochs} - Loss: {loss_val:.4f} - Val Acc: {acc:.2f}%")
    return w, b



from keras.datasets import fashion_mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import wandb

from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def sweep_train():
    from keras.datasets import fashion_mnist
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder

    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    X_train_full = X_train_full.reshape(-1, 28*28) / 255.0
    X_test = X_test.reshape(-1, 28*28) / 255.0

    encoder = OneHotEncoder(sparse_output=False)
    y_train_full_enc = encoder.fit_transform(y_train_full.reshape(-1, 1))

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full_enc, test_size=0.1, random_state=42)

    wandb.init(project='Deep Learning', entity='rharini1423')
    config = wandb.config
    wandb.run.name = f"hl_{config.num_hidden_layers}_bs_{config.batch_size}_ac_{config.activation}_loss_{config.loss_fn}"
    wandb.config.update({'loss_fn': config.loss_fn})


    hidden = [config.hidden_layer_size] * config.num_hidden_layers
    layer_sizes = [784] + hidden + [10]

    model = train(
          X_train, y_train,
          X_val, y_val,
          sizes=layer_sizes,
          opt_name=config.optimizer,
          epochs=config.epochs,
          bs=config.batch_size,
          lr=config.learning_rate,
          wd=config.weight_decay,
          init=config.init_method,
          act=config.activation,
          loss_fn=config.loss_fn
    )


    _, test_acts = fwd(X_test, model[0], model[1], config.activation)

    y_test_preds = test_acts[-1]

    y_test_labels = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
    y_test_preds_labels = np.argmax(y_test_preds, axis=1)

    test_acc = accuracy_score(y_test_labels, y_test_preds_labels)
    wandb.log({'test_accuracy': test_acc})

    cm = confusion_matrix(y_test_labels, y_test_preds_labels)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.title("Normalized Confusion Matrix - Fashion MNIST")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()

    wandb.finish()


sweep_config = {
    'method': 'random',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'epochs': {'values': [5, 10]},
        'num_hidden_layers': {'values': [3, 4, 5]},
        'hidden_layer_size': {'values': [32, 64, 128]},
        'weight_decay': {'values': [0, 0.0005, 0.5]},
        'learning_rate': {'values': [1e-3, 1e-4]},
        'optimizer': {'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']},
        'batch_size': {'values': [16, 32, 64]},
        'init_method': {'values': ['random', 'xavier']},
        'activation': {'values': ['sigmoid', 'tanh', 'relu']},
        'loss_fn': {'values': ['cross_entropy', 'squared']}
    }
}


sweep_id = wandb.sweep(sweep_config, project='Deep Learning')
wandb.agent(sweep_id, function=sweep_train,count=10)
