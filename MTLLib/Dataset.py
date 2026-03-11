import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def load_dataset(split="train"):
    dataset_dir = os.getenv("TOXICITY_DATASET_DIR", "./Dataset/jigsaw-unintended-bias-in-toxicity-classification")
    if split == "train":
        clean_train_path = os.path.join(dataset_dir, "balanced_train.csv")
        clean_valid_path = os.path.join(dataset_dir, "balanced_valid.csv")
        
        if os.path.exists(clean_train_path) and os.path.exists(clean_valid_path):
            logger.info(f"Existing balanced datasets found in {dataset_dir}. Loading them directly...")
            balanced_train_df = pd.read_csv(clean_train_path)
            balanced_valid_df = pd.read_csv(clean_valid_path)
            logger.info(f"Loaded {len(balanced_train_df)} training samples and {len(balanced_valid_df)} validation samples from cache.")
            return balanced_train_df, balanced_valid_df
            
        raw_path = os.path.join(dataset_dir, "train.csv")
        nrows = 100000

    elif split == "test":
        clean_test_path = os.path.join(dataset_dir, "cleaned_test.csv")
        if os.path.exists(clean_test_path):
            logger.info(f"Existing cleaned test dataset found in {dataset_dir}. Loading it directly...")
            test_df = pd.read_csv(clean_test_path)
            logger.info(f"Loaded {len(test_df)} test samples from cache.")
            return test_df
            
        raw_path = os.path.join(dataset_dir, "test.csv")
        nrows = 10000
    
    else:
        raise ValueError(f"Invalid split name '{split}'. Supported splits are 'train' or 'test'.")

    logger.info(f"Loading first 2 rows of '{raw_path}' to infer datatypes...")
    temp_dataset_df = pd.read_csv(raw_path, nrows=2, low_memory=True, memory_map=True)
    
    mapdtype = {'int64': 'int32', 'float64':'float32'}
    dataset_dtypes = list(temp_dataset_df.dtypes.apply(str).replace(mapdtype))
    dataset_dtypes = {key: value for (key, value) in enumerate(dataset_dtypes)}

    logger.info(f"Loading {nrows} rows from '{raw_path}' with explicit datatypes...")
    dataset_df = pd.read_csv(raw_path, nrows=nrows, low_memory=True, memory_map=True, dtype=dataset_dtypes)

    if split == "test":
        logger.info(f"Saving cleaned test dataset to {dataset_dir}...")
        dataset_df.to_csv(clean_test_path, index=False)
        return dataset_df

    logger.info(f"Loaded {len(dataset_df)} rows. Filtering toxic vs safe comments...")
    toxic_df = dataset_df[dataset_df['target'] >= 0.5]
    safe_df = dataset_df[dataset_df['target'] < 0.5]

    # 2. Undersample the safe comments to match the number of toxic comments
    logger.info(f"Undersampling safe comments ({len(safe_df)}) to match toxic comments ({len(toxic_df)})...")
    safe_undersampled_df = safe_df.sample(n=len(toxic_df), random_state=42)

    # 3. Recombine and shuffle
    logger.info("Recombining and shuffling the balanced dataset...")
    balanced_train_df = pd.concat([toxic_df, safe_undersampled_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    balanced_valid_df = balanced_train_df[:2000]
    balanced_train_df = balanced_train_df[2000:]
    
    logger.info(f"Dataset preparation complete: {len(balanced_train_df)} training samples, {len(balanced_valid_df)} validation samples.")
    
    logger.info(f"Saving balanced datasets to {dataset_dir}...")
    balanced_train_df.to_csv(clean_train_path, index=False)
    balanced_valid_df.to_csv(clean_valid_path, index=False)
    
    return balanced_train_df, balanced_valid_df


## Writing a custom Dataset class for the Toxicity Dataset
class ToxicityDataset(Dataset):
    """PyTorch Class for Toxicity Dataset"""
    def __init__(self, toxicity_df, embedding_model, train=True):
        """
        Args
            toxicity_df: The input DataFrame
            embedding_model: The SentenceTransformer model to be used for embeddings
        Returns
            PyTorch Dataset object
        """
        logger.info(f"Initializing ToxicityDataset (train={train}) with {len(toxicity_df)} samples...")
        self.toxicity_df = toxicity_df
        self.embedding_model = embedding_model
        self.train = train

        if self.train:
            self.toxicity_df.loc[:, 'toxic_label'] = np.where(self.toxicity_df['target'] >= 0.5, 1, 0)
            self.toxicity_df.loc[:, 'click_label'] = np.where(self.toxicity_df['likes'] >= 2, 1, 0)

        comment_text = self.toxicity_df['comment_text'].tolist()
        logger.info(f"Encoding {len(comment_text)} comments using embedding model... This may take a while.")
        self.comment_text_embd = embedding_model.encode(comment_text)
        logger.info("Encoding complete.")

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.toxicity_df)
        
    def __getitem__(self, idx):
        """
        Args
            idx: index of the item to be retrived
        Returns
            embeddings, binary toxic label, binary click label at the idx
        """
        df_idx = self.toxicity_df.iloc[idx]
        
        if self.train:
            out = (self.comment_text_embd[idx], df_idx['toxic_label'], df_idx['click_label'])
        else: 
            out = (self.comment_text_embd[idx])
            
        return out