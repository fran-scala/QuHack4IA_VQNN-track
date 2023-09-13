# QuHack4IA_VQNN-track Repository

This repository contains code and data related to the QuHack4IA– Quantum Hackathon for Industrial Applications (12-13 September 2023), Variational Quantum Neural Network (VQNN) track project.

Concrete compressive strength is the measure of the ability to withstand axial loads without failing and is typically expressed in megapascals (MPa). Achieving the desired compressive strength is crucial in ensuring the structural integrity and safety of concrete-based structures.

Here we train multiple VQNNs to learn how different features determine the concrete compressive strength. This is done by analysing a dataset of 1030 instances with 8 different features.

In addition developed a demo of ***COMPRESS BOT*** a chatbot able to predict the concrete compressive strength from given ingredients.

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
- [datapreprocessing](#datapreprocessing): Contains data preprocessing code and notes.
- [model](#model): Contains VQNN model implementations.
- [model_selection](#model_selection): Hyperparameter tuning files.
- [prediction](#prediction): Classical predictions.
- [demo](#demo): demo of COMPRESS BOT.
- [plots](#plots): Contains various plots and visualizations.
- [utils](#utils): Contains utility code for plotting results.

### dataset
Contains dataset files in various formats.

- `concrete_data.csv`: Raw dataset file.
- `dataset_with_outliers.csv`: Dataset file with outliers.
- `dataset_without_outliers.csv`: Dataset file without outliers.
- `dataset_without_outliers_without_feature.csv`: Dataset file without outliers and a specific feature.

### advanced
Contains advanced VQNN implementations.

- `VQNN_basic_entangler.py`: Implementation of a basic VQNN with an entangler circuit.
- `basic_entangler_circuit.pdf`: PDF documentation for the basic entangler circuit.
- `VQNN_random_ansatz.py`: Implementation of a VQNN with a random ansatz circuit.
- `random_circuit.pdf`: PDF documentation for the random ansatz circuit.

### datapreprocessing
Contains data preprocessing code and notes.

- `datapreprocessing.ipynb`: Jupyter Notebook for data preprocessing.
- `notes.md`: Notes related to data preprocessing.

### model
Contains VQNN model implementations.

- `VQNN_linear.ipynb`: Jupyter Notebook for linear VQNN.
- `VQNN_linear.py`: Python script for linear VQNN.
- `VQNN_nonlinear.ipynb`: Jupyter Notebook for nonlinear VQNN.
- `VQNN_nonlinear.py`: Python script for nonlinear VQNN.

### demo

- `banner.txt`: Banner image in ASCII.
- `demo.py`: A demo of COMPRESS BOT a chatbot able to predict the concrete compressive strength from given ingredients.


### plots
Contains various plots and visualizations.

### utils
Contains utility code for plotting results.

- `plot_results.ipynb`: Jupyter Notebook for plotting results.

## Requirements

- See `requirements.txt` for the Python library requirements for running the code in this repository.



