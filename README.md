# Advance Multi-Task Learning

## Overview
Advance Multi-Task Learning is a PyTorch-based project focused on exploring and implementing various Multi-Task Learning (MTL) architectures. The current implementation demonstrates **Shared-Bottom**, **Mixture-of-Experts (MoE)**, and **Multi-gate Mixture-of-Experts (MMoE)** models designed to simultaneously predict two tasks:
1. **Engagement** (Predicting click probability)
2. **Toxicity** (Predicting if a comment is toxic or not)

The project leverages `sentence-transformers` for generating comment text embeddings to serve as input feature representations for the PyTorch models.

## Project Structure
The repository is structured to organize datasets, model architectures, and training utilities logically.

### 📁 MTLLib
This package contains core utilities for data loading and model training.
- **`Dataset.py`**: Defines the `ToxicityDataset` PyTorch class. It handles loading data (using chunking and predefined dtypes for efficiency), uses a Sentence Transformer model to encode text comments into embeddings, and serves batches with inputs, binary valid toxic labels, and simulated click labels. Incorporates execution logging.
- **`ModelTrainer.py`**: Contains the `ModelTrainer` class. It abstracts the training loops, encapsulates loss calculations via Binary Cross-Entropy (BCE) for multiple tasks, evaluates validation data iteratively, and automatically saves the best-performing model weights dynamically per architecture.

### 📁 Models
Stores the PyTorch model architectures built for experimenting with Multi-Task Learning.
- **`SharedBottomModel.py`**: Implements the `SharedBottomMLT` architecture. It takes encoded dimension arrays (384 for MiniLM-L6) and passes them through a shared linear layer before branching off into two distinct task-specific heads (Engagement Head and Toxicity Head), each mapping to a single probability via Sigmoid activation.
- **`MoE.py`**: Implements the `BasicMoE` architecture, a Mixture-of-Experts model using a gating network to weigh expert predictions. It allows configuring the number of experts and adding pre-trained experts.
- **`MMoE.py`**: Implements the `BasicMMoE` architecture, a Multi-gate Mixture-of-Experts model assigning separate gating networks specifically optimized for each task.

### 📁 Notebooks
- **`Multi-Task Learning Notebook.ipynb`**: An extensive Jupyter Notebook mapping out the exploratory process. It includes data preparation steps using pandas, PyTorch Dataset definition, full exploratory Shared Bottom, MoE, and MMoE model training workflows, and evaluation functions that utilize `scikit-learn` to plot Precision-Recall and ROC-AUC curves for assessing task performance metrics visually.

### Root Files
- **`train.py`**: The main training script. It loads environment variables, initializes the `SentenceTransformer` and device setups, prepares DataLoaders, and sequentially trains the Shared-Bottom, MoE, and MMoE architectures. The best model weights across experiments are saved in `ModelBinaries`. 
- **`test.py`**: The model evaluation and inference script. It dynamically loads previously trained models from `ModelBinaries`, calculates performance metrics (`ROC-AUC`, `Loss`, `Accuracy`) using a labeled validation subset, and generates pure inferences on an unlabeled true test set. Predictions are nicely formatted into a `test_predictions.md` artifact.
- **`.env`**: Holds local environment variables, primarily the root path to the dataset directory (`TOXICITY_DATASET_DIR`).
- **`pyproject.toml`**: Metadata and dependency file outlining the `advance-multi-task-learning` project's configuration for Python 3.12+. Managed via `uv`.

## Installation & Setup

1. **Clone the repository** and navigate to the root directory.
2. Ensure you have the [uv](https://github.com/astral-sh/uv) package manager installed.
3. Sync the environment and install dependencies:
    ```bash
    uv sync
    ```
4. **Environment Configuration**: Ensure your `.env` file exists at the root with the dataset directory path:
    ```env
    TOXICITY_DATASET_DIR="./Dataset/jigsaw-unintended-bias-in-toxicity-classification"
    ```
    > Note: On its first run, `MTLLib.Dataset` will parse the raw `.csv` files into subsets and permanently cache them as `balanced_train.csv`, `balanced_valid.csv`, and `cleaned_test.csv` in this directory to drastically speed up future loads.

## Usage

To start training experiments evaluating the Shared-Bottom, MoE, and MMoE architectures, simply execute:
```bash
uv run train.py
```
This will automatically download the `all-MiniLM-L6-v2` transformer model, process textual inputs into dense embeddings on the fly, and save the best model weights dynamically.

Once trained, to evaluate all available models and generate predictions, run:
```bash
uv run test.py
```
This will dynamically find the architectures that have weights in `ModelBinaries`, output evaluation metrics to the console, and generate a markdown table sheet with pure dataset predictions in `test_predictions.md`.
