import torch
import os
import logging
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from MTLLib.Dataset import load_dataset, ToxicityDataset
from Models.MMoE import BasicMMoE
from Models.MoE import BasicMoE
from Models.SharedBottomModel import SharedBottomMLT
from MTLLib.ModelTrainer import ModelTrainer
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from MTLLib.MTLLoss import MultiTaskLoss

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

load_dotenv()

def SharedModelExperiment(toxicity_train_dataloader, toxicity_valid_dataloader, device):
    logger.info("-" * 50)
    logger.info("Shared Model Experiment")
    logger.info("-" * 50)
    
    shared_bottom_model = SharedBottomMLT()
    shared_bottom_model.to(device)

    is_regression = torch.tensor([False, False])
    criterion = MultiTaskLoss(nn.BCELoss(), nn.BCELoss(), is_regression).to(device)
    optimizer = optim.Adam(list(shared_bottom_model.parameters()) + list(criterion.parameters()), lr=0.001, weight_decay=0.001)

    model_trainer = ModelTrainer(shared_bottom_model, toxicity_train_dataloader, toxicity_valid_dataloader, criterion, optimizer, device, "SharedBottomModel")
    model_trainer.train()

def BasicMoEExperiment(toxicity_train_dataloader, toxicity_valid_dataloader, device):
    logger.info("-" * 50)
    logger.info("BasicMoE Experiment")
    logger.info("-" * 50)
    
    BasicMoE_model = BasicMoE(384, 128, 3)
    BasicMoE_model.to(device)

    is_regression = torch.tensor([False, False])
    criterion = MultiTaskLoss(nn.BCELoss(), nn.BCELoss(), is_regression).to(device)
    optimizer = optim.Adam(list(BasicMoE_model.parameters()) + list(criterion.parameters()), lr=0.001, weight_decay=0.001)

    model_trainer = ModelTrainer(BasicMoE_model, toxicity_train_dataloader, toxicity_valid_dataloader, criterion, optimizer, device, "BasicMoE")
    model_trainer.train()

# def TrainedExpertsMoEExperiment(toxicity_train_dataloader, toxicity_valid_dataloader, device):
#     logger.info("-" * 50)
#     logger.info("TrainedExpertsMoE Experiment")
#     logger.info("-" * 50)
    
#     TrainedExpertsMoE_model = BasicMoE(trained_experts)
#     TrainedExpertsMoE_model.to(device)

#     criterion = [nn.BCELoss(), nn.BCELoss()]
#     optimizer = optim.Adam(TrainedExpertsMoE_model.parameters(), lr=0.001, weight_decay=0.001)

#     model_trainer = ModelTrainer(TrainedExpertsMoE_model, toxicity_train_dataloader, toxicity_valid_dataloader, criterion, optimizer, device, "TrainedExpertsMoE")
#     model_trainer.train()

def BasicMMoEExperiment(toxicity_train_dataloader, toxicity_valid_dataloader, device):
    logger.info("-" * 50)
    logger.info("BasicMMoE Experiment")
    logger.info("-" * 50)
    
    BasicMMoE_model = BasicMMoE(384, 128, 3)
    BasicMMoE_model.to(device)
    
    is_regression = torch.tensor([False, False])
    criterion = MultiTaskLoss(nn.BCELoss(), nn.BCELoss(), is_regression).to(device)
    optimizer = optim.Adam(list(BasicMMoE_model.parameters()) + list(criterion.parameters()), lr=0.001, weight_decay=0.001)

    model_trainer = ModelTrainer(BasicMMoE_model, toxicity_train_dataloader, toxicity_valid_dataloader, criterion, optimizer, device, "BasicMMoE")
    model_trainer.train()

def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    ## Loading the train and validation data
    toxicity_train_df, toxicity_valid_df = load_dataset(split="train")
    
    toxicity_train_dataset = ToxicityDataset(toxicity_train_df, embedding_model)
    toxicity_valid_dataset = ToxicityDataset(toxicity_valid_df, embedding_model)
    toxicity_train_dataloader = DataLoader(toxicity_train_dataset, batch_size=32, shuffle=True)
    toxicity_valid_dataloader = DataLoader(toxicity_valid_dataset, batch_size=32, shuffle=False)

    ## Loading the test data
    toxicity_test_df = load_dataset(split="test")
    
    toxicity_test_dataset = ToxicityDataset(toxicity_test_df, embedding_model, train=False)
    toxicity_test_dataloader = DataLoader(toxicity_test_dataset, batch_size=32, shuffle=False)

    ## Model Initializatoins
    SharedModelExperiment(toxicity_train_dataloader, toxicity_valid_dataloader, device)
    BasicMoEExperiment(toxicity_train_dataloader, toxicity_valid_dataloader, device)
    BasicMMoEExperiment(toxicity_train_dataloader, toxicity_valid_dataloader, device)

    
if __name__ == "__main__":
    main()