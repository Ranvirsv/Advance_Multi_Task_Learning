from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

def load_dataset(dataset_path:str):
    temp_dataset_df = pd.read_csv(dataset_path, nrows=2, low_memory=True, memory_map=True)

    mapdtype = {'int64': 'int32', 'float64':'float32'}
    dataset_dtypes = list(temp_dataset_df.dtypes.apply(str).replace(mapdtype))
    dataset_dtypes = {key: value for (key, value) in enumerate(dataset_dtypes)}

    dataset_df = pd.read_csv(dataset_path, nrows=20000, low_memory=True, memory_map=True, dtype=dataset_dtypes)

    return dataset_df


## Writing a custom Dataset class for the Toxicity Dataset
class ToxicityDataset(Dataset):
    """PyTorch Class for Toxicity Dataset"""
    def __init__(self, toxicity_df, embedding_model):
        """
        Args
            toxicity_df: The input DataFrame
            embedding_model: The SentenceTransformer model to be used for embeddings
        Returns
            PyTorch Dataset object  
        """
        self.toxicity_df = toxicity_df
        self.embedding_model = embedding_model
        
        self.toxicity_df['toxic_label'] = np.where(self.toxicity_df['target'] >= 0.5, 1, 0)
        self.toxicity_df['click_label'] = np.random.randint(2, size=20000)

        comment_text = self.toxicity_df['comment_text'].tolist()
        self.comment_text_embd = embedding_model.encode(comment_text)

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.toxicity_df)
        
    def __get_item__(self, idx):
        """
        Args
            idx: index of the item to be retrived
        Returns
            embeddings, binary toxic label, binary click label at the idx
        """
        df_idx = self.toxicity_df.iloc[idx]
        return (self.comment_text_embd[idx], df_idx['toxic_label'], df_idx['click_label'])
        