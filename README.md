# Advance Multi-Task Learning

## Overview
Advance Multi-Task Learning is a PyTorch-based project focused on exploring and implementing various Multi-Task Learning (MTL) architectures. The current implementation demonstrates a **Shared-Bottom Model** designed to simultaneously predict two tasks:
1. **Engagement** (Predicting click probability)
2. **Toxicity** (Predicting if a comment is toxic or not)

The project leverages `sentence-transformers` for generating comment text embeddings to serve as input feature representations for the PyTorch models.

## Project Structure
The repository is structured to organize datasets, model architectures, and training utilities logically.

### 📁 MLTLib
This package contains core utilities for data loading and model training.
- **`Dataset.py`**: Defines the `ToxicityDataset` PyTorch class. It handles loading data (using chunking and predefined dtypes for efficiency), uses a Sentence Transformer model to encode text comments into embeddings, and serves batches with inputs, binary valid toxic labels, and simulated click labels.
- **`ModelTrainer.py`**: Contains the `ModelTrainer` class. It abstracts the training loops, encapsulates loss calculations via Binary Cross-Entropy (BCE) for multiple tasks, evaluates validation data iteratively, and automatically saves the best-performing model weights (`best_model.pth`).

### 📁 Models
Stores the PyTorch model architectures built for experimenting with Multi-Task Learning.
- **`SharedBottomModel.py`**: Implements the `SharedBottomMLT` architecture. It takes encoded dimension arrays (384 for MiniLM-L6) and passes them through a shared linear layer before branching off into two distinct task-specific heads (Engagement Head and Toxicity Head), each mapping to a single probability via Sigmoid activation.
- **`MoE.py`**: A foundational placeholder class for implementing the Mixture-of-Experts architecture in upcoming iterations.
- **`MMoE.py`**: A foundational placeholder class for implementing the Multi-gate Mixture-of-Experts architecture in upcoming iterations.

### 📁 Notebooks
- **`Multi-Task Learning Notebook.ipynb`**: An extensive Jupyter Notebook mapping out the exploratory process. It includes data preparation steps using pandas, PyTorch Dataset definition, full exploratory Shared Bottom model training workflows, and evaluation functions that utilize `scikit-learn` to plot Precision-Recall and ROC-AUC curves for assessing task performance metrics visually.

### Root Files
- **`main.py`**: The main execution script. It loads environment variables, initializes the `SentenceTransformer` and sets device mappings, prepares DataLoaders for both train and validation datasets, and then initiates the Multi-Task training experiment using the definitions from `Models` and `MLTLib`.
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

To start a training experiment using the Shared-Bottom architecture out-of-the-box, execute:
```bash
python main.py
```

This will automatically trigger downloading/loading the `all-MiniLM-L6-v2` transformer model, process your text into dense embeddings on the fly, and run the training pipeline over your designated dataset. The best weights observed will be generated locally as `best_model.pth`.
