# Neural Network from Scratch in Python

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
5. [Testing](#testing)


## Introduction
This project is a Python im implementation of a Neural Network built from scratch. I created this as a learning project to give me a deeper understanding of the inner workings of neural networks by implementing one without the use of high-level libraries.

The project includes a Jupyter notebook, `Colab_Housing_Test.ipynb`, which can be run in Google Colab to test the neural network on a housing dataset. The dataset is included in the `data` directory.

The neural network itself is implemented in the `neural_network` directory, which includes separate Python files for different components of the network such as activation functions (`activation.py`), cost functions (`cost.py`), layers (`layers.py`), the network structure (`network.py`), and optimization algorithms (`optimizer.py`).

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/ErikFolkesson/nn_from_scratch_python.git
    ```

2. Navigate to the project directory:
    ```bash
    cd nn_from_scratch_python
    ```

3. Install the dependencies from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Testing
To test the project, you can use the `Colab_Housing_Test.ipynb` notebook. This notebook is designed to be run in Google Colab. Here are the steps to run the notebook:

1. Open the `Colab_Housing_Test.ipynb` notebook in your browser.

2. Click on the "Open in Colab" button at the top of the notebook. 

3. Once the notebook is open in Google Colab, you can run it by clicking on the "Runtime" menu and then "Run all". This will execute all the cells in the notebook.