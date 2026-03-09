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
- **`main.py`**: The main execution script. It loads environment variables, initializes the `SentenceTransformer` and device setups, prepares DataLoaders for the train, valid, and test datasets, and then sequentially runs training experiments for Shared-Bottom, MoE, and MMoE architectures using components from `Models` and `MTLLib`.
- **`.env`**: Holds local environment variables, primarily the paths to the dataset splits. Needs to include variables like `TOXICITY_TRAIN_PATH` and `TOXICITY_TEST_PATH`.
- **`pyproject.toml`**: Metadata and dependency file outlining the `advance-multi-task-learning` project's configuration for Python 3.12+.

## Installation & Setup

1. **Clone the repository** and navigate to the root directory.
2. **Ensure Python 3.12+** is installed.
3. Install required libraries (from the exploratory notebook's footprint): 
    ```bash
    pip install torch pandas numpy scikit-learn sentence-transformers matplotlib python-dotenv
    ```
4. **Environment Configuration**: Ensure your `.env` file exists at the root with dataset paths:
    ```env
    TOXICITY_TRAIN_PATH="/path/to/your/train.csv"
    TOXICITY_TEST_PATH="/path/to/your/test.csv"
    ```

## Usage

To start training experiments evaluating the Shared-Bottom, MoE, and MMoE architectures sequentially, execute:
```bash
python main.py
```

This will automatically trigger downloading/loading the `all-MiniLM-L6-v2` transformer model, process textual inputs into dense embeddings on the fly, and run the entire sequential training pipeline over your datasets. The best model weights across experiments are observed and saved automatically within a dynamically generated `ModelBinaries` directory, alongside individual training logs.
