# QuHack4IA_VQNN-track Repository

This repository contains code and data related to the QuHack4IA– Quantum Hackathon for Industrial Applications (12-13 September 2023), Variational Quantum Neural Network (VQNN) track project. The team #4 members are: Lorenzo Bergadano,Alessandro Danesin, Giorgia Mazzaro, Francesco Scala.

Concrete compressive strength is the measure of the ability to withstand axial loads without failing and is typically expressed in megapascals (MPa). Achieving the desired compressive strength is crucial in ensuring the structural integrity and safety of concrete-based structures.

Here we train multiple VQNNs to learn how different features determine the concrete compressive strength. This is done by analysing a dataset of 1030 instances with 8 different features.

In addition, we developed a demo of ***COMPRESS BOT*** a chatbot able to predict the concrete compressive strength from given ingredients.

-------------------------

## Table of content
- [Cloning the repo](#cloning-the-repo)
- [Requirements](#requirements)
- [Execution](#execution)
- [Device Comparison](#device-comparison)
  - [Classical approach](#classical-approach)
  - [Quantum approach](#quantum-approach)
- [Directory Structure](#directory-structure)
  - [Dataset](#dataset)
  - [Advanced tasks](#advanced)
  - [Data preprocessing](#data-preprocessing)
  - [Model](#Model)
  - [Demo](#Demo)
  - [Plots](#Plots)
  - [utils](#utils)
- [Contacts](#contacts)

------------------

## Cloning the repo
To clone the repo through HTTPS or SSH, you must have installed Git on your operating system.<br>
Then you can open a new terminal and type the following command (this is the cloning through HTTPS):
```bash
    git clone https://github.com/fran-scala/QuHack4IA_VQNN-track.git
```

If you don't have installed Git, you can simply download the repository by pressing <i>"Download ZIP"</i>.

## Requirements

See `requirements.txt` for the Python library requirements for running the code in this repository.
Once the repo is cloned, some Python libraries are required to properly set up your (virtual) environment.


They can be installed via pip:
```bash
    pip install -r requirements.txt
```

or via conda:
```bash
    conda create --name <env_name> --file requirements.txt
```
-----------------------
## Execution

The `/demo/demo.py` is a demonstrator of our model: you can run it after training the model and produce the results
(inside the results folder).<br>
So you can execute the command `python ./demo/demo.py` to run the demo.

To run the training of the model, you can run:
- `python ./model/VQNN_linear.py`: it executes the model by using the Angle Encoding with the rotation angle computed as 
  $\pi x$
- `python ./model/VQNN_nonlinear.py`: it executes the model by using the Angle Encoding with the rotation angle computed as
  $2arctan(x)$

On the other hand, you can use the jupyter notebook present in the same folder.

------------------------

## Device Comparison
We compute the Mean Square Error (MSE) for both the below approaches.
### Classical approach
- Normalized dataset:
  - MSE: 0.014592172783657214
- Normalized dataset with features reduction:
  - MSE: 0.014217871731188965
- Normalized dataset with features reduction and without outliers:
  - MSE: 0.014926592236208975
### Quantum approach
- Normalized dataset:
  - MSE: 0.10057372
- Normalized dataset with features reduction:
  - MSE: 0.039087
- Normalized dataset with features reduction and without outliers:
  - MSE: 0.043391567


## Directory Structure

- [dataset](#dataset): Contains dataset files in various formats.
- [advanced](#advanced): Contains VQNN implementations for the advanced task.
- [datapreprocessing](#data-preprocessing): Contains data preprocessing code and notes.
- [model](#model): Contains VQNN model implementations.
- [model_selection](#model-selection): Hyperparameter tuning files.
- [prediction](#prediction): Classical predictions.
- [demo](#demo): demo of COMPRESS BOT.
- [plots](#plots): Contains various plots and visualizations.
- [utils](#utils): Contains utility code for plotting results.

### Dataset
Contains dataset files in various formats.

- `concrete_data.csv`: Raw dataset file.
- `dataset_with_outliers.csv`: Dataset file with outliers.
- `dataset_without_outliers.csv`: Dataset file without outliers.
- `dataset_without_outliers_without_feature.csv`: Dataset file without outliers and a specific feature.

The original dataset is public on Kaggle: https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set

### Advanced
Contains advanced VQNN implementations.

- `VQNN_basic_entangler.py`: Implementation of a basic VQNN with an entangler circuit.
- `basic_entangler_circuit.pdf`: PDF documentation for the basic entangler circuit.
- `VQNN_random_ansatz.py`: Implementation of a VQNN with a random ansatz circuit.
- `random_circuit.pdf`: PDF documentation for the random ansatz circuit.

### Data preprocessing
Contains data preprocessing code and notes.

- `datapreprocessing.ipynb`: Jupyter Notebook for data preprocessing.
- `notes.md`: Notes related to data preprocessing.

### Model
Contains VQNN model implementations.

- `VQNN_linear.ipynb`: Jupyter Notebook for linear VQNN.
- `VQNN_linear.py`: Python script for linear VQNN.
- `VQNN_nonlinear.ipynb`: Jupyter Notebook for nonlinear VQNN.
- `VQNN_nonlinear.py`: Python script for nonlinear VQNN.

### Model Selection
Contains VQNN hyperaparameters tuning and the best model.

### Demo

- `banner.txt`: Banner image in ASCII.
- `demo.py`: A demo of COMPRESS BOT a chatbot able to predict the concrete compressive strength from given ingredients.

### Plots
Contains various plots and visualizations.


### utils
Contains utility code for plotting results.

- `plot_results.ipynb`: Jupyter Notebook for plotting results.

-------------------------------------------------------------

## Contacts

| Author                 | GitHub                                     | 
|------------------------|--------------------------------------------|
| **Lorenzo Bergadano**  | [lolloberga](https://github.com/lolloberga) |
| **Alessandro Danesin** | [ale100gs](https://github.com/ale100gs)    |
| **Giorgia Mazzaro**    |      |
| **Francesco Scala**    | [fran-scala](https://github.com/fran-scala) |
