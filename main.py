import torch
from Dataset import load_dataset, ToxicityDataset
from MMoE import MMoE
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

## File to get the data, and run models 

def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    toxicity_train_path = os.getenv("TOXICITY_TRAIN_PATH")
    toxicity_train_df = load_dataset(toxicity_train_path)
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    toxicity_train_dataset = ToxicityDataset(toxicity_train_df, embedding_model)
    toxicity_train_dataloader = DataLoader(toxicity_train_dataset, batch_size=32, shuffle=True)

    toxicity_test_path = os.getenv("TOXICITY_TEST_PATH")
    toxicity_test_df = load_dataset(toxicity_test_path)
    
    toxicity_test_dataset = ToxicityDataset(toxicity_test_df, embedding_model)
    toxicity_test_dataloader = DataLoader(toxicity_test_dataset, batch_size=32, shuffle=True)

    mmoe_model = MMoE()
    mmoe_model.to(device)
    
    
if __name__ == "__main__":
    main()