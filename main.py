import torch
from MTLLib.Dataset import load_dataset, ToxicityDataset
from Models.MMoE import BasicMMoE
from Models.MoE import BasicMoE
from Models.SharedBottomModel import SharedBottomMLT
from MTLLib.ModelTrainer import ModelTrainer
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

def SharedModelExperiment(toxicity_train_dataloader, toxicity_valid_dataloader, device):
    shared_bottom_model = SharedBottomMLT()
    shared_bottom_model.to(device)

    criterion = [nn.BCELoss(), nn.BCELoss()]
    optimizer = optim.Adam(shared_bottom_model.parameters(), lr=0.001, weight_decay=0.001)

    model_trainer = ModelTrainer(shared_bottom_model, toxicity_train_dataloader, toxicity_valid_dataloader, criterion, optimizer, device, "SharedBottomModel")
    model_trainer.train()

def MoEExperiment(toxicity_train_dataloader, toxicity_valid_dataloader, device):
    BasicMoE_model = BasicMoE()
    BasicMoE_model.to(device)

    optimizer = optim.Adam(BasicMoE_model.parameters(), lr=0.001, weight_decay=0.001)

    model_trainer = ModelTrainer(BasicMoE_model, toxicity_train_dataloader, toxicity_valid_dataloader, criterion, optimizer, device, "BasicMoE")
    model_trainer.train()

def MMoEExperiment(toxicity_train_dataloader, toxicity_valid_dataloader, device):
    BasicMMoE_model = BasicMMoE()
    BasicMMoE_model.to(device)

    optimizer = optim.Adam(BasicMMoE_model.parameters(), lr=0.001, weight_decay=0.001)

    model_trainer = ModelTrainer(BasicMMoE_model, toxicity_train_dataloader, toxicity_valid_dataloader, criterion, optimizer, device, "BasicMMoE")
    model_trainer.train()

def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    ## Loading the train and validation data
    toxicity_train_path = os.getenv("TOXICITY_TRAIN_PATH")
    toxicity_train_df = load_dataset(toxicity_train_path)
    toxicity_valid_df = toxicity_train_df[:5000]
    toxicity_train_df = toxicity_train_df[5001:]
    
    toxicity_train_dataset = ToxicityDataset(toxicity_train_df, embedding_model)
    toxicity_valid_dataset = ToxicityDataset(toxicity_valid_df, embedding_model)
    toxicity_train_dataloader = DataLoader(toxicity_train_dataset, batch_size=32, shuffle=True)
    toxicity_valid_dataloader = DataLoader(toxicity_valid_dataset, batch_size=32, shuffle=False)

    ## Loading the test data
    toxicity_test_path = os.getenv("TOXICITY_TEST_PATH")
    toxicity_test_df = load_dataset(toxicity_test_path)
    
    toxicity_test_dataset = ToxicityDataset(toxicity_test_df, embedding_model)
    toxicity_test_dataloader = DataLoader(toxicity_test_dataset, batch_size=32, shuffle=False)

    ## Model Initializatoins
    SharedModelExperiment(toxicity_train_dataloader, toxicity_valid_dataloader, device)
    MoEExperiment(toxicity_train_dataloader, toxicity_valid_dataloader, device)
    MMoEExperiment(toxicity_train_dataloader, toxicity_valid_dataloader, device)

    
if __name__ == "__main__":
    main()