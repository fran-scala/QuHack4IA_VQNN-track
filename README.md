# QuHack4IA_VQNN-track Repository

This repository contains code and data related to the QuHack4IA `when-where-what` , Variational Quantum Neural Network (VQNN) track project.

Concrete compressive strength is the measure of the ability to withstand axial loads without failing and is typically expressed in megapascals (MPa). Achieving the desired compressive strength is crucial in ensuring the structural integrity and safety of concrete-based structures.

Here we train multiple VQNNs to learn how different features determine the concrete compressive strength. This is done by analysing a dataset of 1030 instances with 8 different features.


## Directory Structure

- [dataset](#dataset)
- [advanced](#advanced)
- [datapreprocessing](#datapreprocessing)
- [model](#model)
- [plots](#plots)
- [utils](#utils)

### dataset
Contains dataset files in various formats.

- `concrete_data.csv`: Raw dataset file.
- `dataset_with_outliers.csv`: Dataset file with outliers.
- `dataset_without_outliers.csv`: Dataset file without outliers.
- `dataset_without_outliers_without_feature.csv`: Dataset file without outliers and specific features.

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

### Device Comparison
We compute the Mean Square Error (MSE) and the Adjusted-R^2 (Adj-R2) as an accuracy metrics for both the below approaches.
#### Classical approach
- Normalized dataset:
  - MSE: 0.018633318309299648
  - Adj-R2: 0.8856312515340603 
- Normalized dataset without outliers:
  - MSE: 0.015269465774461023
  - Adj-R2: 0.9071801646839951
- Normalized dataset with features reduction:
  - MSE: 0.01824797554399269
  - Adj-R2: 0.8884444467700594
- Normalized dataset with features reduction and without outliers:
  - MSE: 0.015918447676270243
  - Adj-R2: 0.903665207748923
#### Quantum approach
- Normalized dataset with features reduction:
- Normalized dataset with features reduction and without outliers:
### plots
Contains various plots and visualizations.

#### linear
Plots related to linear VQNN.

- `with_outliers_compare_mse.pdf`: Comparison plot of mean squared error (MSE) with outliers.
- `with_outliers_compare_mse_per_epoch.pdf`: Comparison plot of MSE per epoch with outliers.

#### nonlinear
Plots related to nonlinear VQNN.

- `no_outliers_compare_mse.pdf`: Comparison plot of MSE without outliers.
- `no_outliers_compare_mse_per_epoch.pdf`: Comparison plot of MSE per epoch without outliers.

### utils
Contains utility code for plotting results.

- `plot_results.ipynb`: Jupyter Notebook for plotting results.

## Requirements

- See `requirements.txt` for the Python library requirements for running the code in this repository.

## Demo

- `demo.txt`: A demo or usage guide can be found here.

Feel free to explore the subdirectories for more details on each component of the project.
