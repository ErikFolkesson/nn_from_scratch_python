# Neural Network from Scratch in Python

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
5. [Testing](#testing)


## Installation
This project is a Python implementation of a Neural Network built from scratch. It is designed to provide a deeper understanding of the inner workings of neural networks by implementing the fundamental components without the use of high-level libraries.

The project includes a Jupyter notebook, `Colab_Housing_Test.ipynb`, which can be run in Google Colab to test the neural network on a housing dataset. The dataset is included in the `data` directory.

The neural network itself is implemented in the `neural_network` directory, which includes separate Python files for different components of the network such as activation functions (`activation.py`), cost functions (`cost.py`), layers (`layers.py`), the network structure (`network.py`), and optimization algorithms (`optimizer.py`).

The `local_notebooks` directory contains a `testing.ipynb` notebook for local testing, and the `requirements.txt` file lists the Python dependencies required to run the project.

The project is licensed under the MIT License, as detailed in the `LICENSE` file.

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

Now you have successfully installed the project and its dependencies.

## Testing
To test the project, you can use the `Colab_Housing_Test.ipynb` notebook. This notebook is designed to be run in Google Colab, a platform that allows you to run Jupyter Notebooks in the cloud. Here are the steps to run the notebook:

1. Open the `Colab_Housing_Test.ipynb` notebook in your browser. You can find it in the root directory of the project.

2. Click on the "Open in Colab" button at the top of the notebook. This will open the notebook in Google Colab.

3. Once the notebook is open in Google Colab, you can run it by clicking on the "Runtime" menu and then "Run all". This will execute all the cells in the notebook, running the tests for the project.

Please note that running the notebook may take some time, as it includes training a neural network on a housing dataset. You can monitor the progress of the training in the output of the cells.