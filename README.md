# Neural Network from Scratch in Python

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Data](#data)
4. [Testing](#testing)


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

## Data
The project uses the California Housing dataset (`data/housing.csv`) for demonstrating the neural network's regression capabilities. This dataset contains information about housing districts in California derived from the 1990 U.S. Census.

### Dataset Overview
- **Rows**: 20,640 housing districts
- **Features**: 10 columns including geographical, demographic, and housing characteristics
- **Target Variable**: `median_house_value` (median house value for households within a block)

### Dataset Features
| Feature | Description |
|---------|-------------|
| `longitude` | A measure of how far west a house is (higher = more west) |
| `latitude` | A measure of how far north a house is (higher = more north) |
| `housing_median_age` | Median age of houses within a block (higher = older) |
| `total_rooms` | Total number of rooms within a block |
| `total_bedrooms` | Total number of bedrooms within a block |
| `population` | Total number of people residing within a block |
| `households` | Total number of households within a block |
| `median_income` | Median income for households within a block (in tens of thousands USD) |
| `median_house_value` | Median house value for households within a block (in USD) |
| `ocean_proximity` | Location of the house relative to ocean/sea (categorical: NEAR BAY, <1H OCEAN, INLAND, NEAR OCEAN, ISLAND) |

### Data Preprocessing
In the provided notebooks, the following preprocessing steps are applied:
- Missing values in `total_bedrooms` (207 rows) are removed using `dropna()`
- Geographic features (`longitude`, `latitude`) and categorical feature (`ocean_proximity`) are dropped for simplicity
- All remaining features are standardized using `StandardScaler` from scikit-learn
- The target variable (`median_house_value`) is also normalized for better training performance

### Usage
The dataset is used to train the neural network for a regression task, predicting median house values based on the district characteristics. The notebooks demonstrate how to load, preprocess, and use this data with the custom neural network implementation.

## Testing
To test the project, you can use the `Colab_Housing_Test.ipynb` notebook. This notebook is designed to be run in Google Colab. Here are the steps to run the notebook:

1. Open the `Colab_Housing_Test.ipynb` notebook in your browser.

2. Click on the "Open in Colab" button at the top of the notebook. 

3. Once the notebook is open in Google Colab, you can run it by clicking on the "Runtime" menu and then "Run all". This will execute all the cells in the notebook.