# CS6910---Assignment-1


## Setup Instructions

### 1. Clone the Repository

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Requirements

Install the necessary Python libraries using `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Basic Command

```bash
python main.py -d mnist -e 10 -b 32 -o adam -lr 0.001 -a ReLU -nhl 2 -sz 128
```

### 🔧 Argument Details

| Flag                     | Description                                   | Example            |
| ------------------------ | --------------------------------------------- | ------------------ |
| `-d`, `--dataset`        | Dataset to use (`mnist` or `fashion_mnist`)   | `-d fashion_mnist` |
| `-e`, `--epochs`         | Number of training epochs                     | `-e 10`            |
| `-b`, `--batch_size`     | Batch size                                    | `-b 32`            |
| `-o`, `--optimizer`      | Optimizer (`adam`, `sgd`, `nadam`, etc.)      | `-o adam`          |
| `-lr`, `--learning_rate` | Learning rate                                 | `-lr 0.001`        |
| `-a`, `--activation`     | Activation function (`ReLU`, `sigmoid`, etc.) | `-a ReLU`          |
| `-nhl`, `--num_layers`   | Number of hidden layers                       | `-nhl 2`           |
| `-sz`, `--hidden_size`   | Size of each hidden layer                     | `-sz 128`          |

### Example Use Case

To train a neural network on **MNIST** for **10 epochs** using:

* **Adam optimizer**
* **ReLU activation**
* **2 hidden layers**, each with **128 units**
* **Learning rate** of 0.001
* **Batch size** of 32

Run:

```bash
python main.py -d mnist -e 10 -b 32 -o adam -lr 0.001 -a ReLU -nhl 2 -sz 128
```

