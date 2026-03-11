import torch
import os
import logging
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import random

from MTLLib.Dataset import load_dataset, ToxicityDataset
from Models.MMoE import BasicMMoE
from Models.MoE import BasicMoE
from Models.SharedBottomModel import SharedBottomMLT

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

load_dotenv()

def evaluate_model(model, test_dataloader, criterion, device, model_name):
    logger.info(f"Evaluating {model_name}...")
    model.eval()
    
    all_toxicity_labels = []
    all_engagement_labels = []
    all_toxicity_preds = []
    all_engagement_preds = []

    running_engagement_loss = 0.0
    running_toxicity_loss = 0.0
    running_total_loss = 0.0

    with torch.no_grad():
        for inputs, toxicity_label, engagement_label in test_dataloader:
            inputs = inputs.to(device)
            toxicity_label = toxicity_label.to(device)
            engagement_label = engagement_label.to(device)

            eng_out, tox_out = model(inputs)

            engagement_loss = criterion[0](eng_out.view(-1), engagement_label.float())
            toxicity_loss = criterion[1](tox_out.view(-1), toxicity_label.float())
            total_loss = engagement_loss + toxicity_loss

            running_engagement_loss += engagement_loss.item()
            running_toxicity_loss += toxicity_loss.item()
            running_total_loss += total_loss.item()

            all_toxicity_labels.extend(toxicity_label.cpu().numpy())
            all_engagement_labels.extend(engagement_label.cpu().numpy())
            all_toxicity_preds.extend(tox_out.view(-1).cpu().numpy())
            all_engagement_preds.extend(eng_out.view(-1).cpu().numpy())

    num_batches = len(test_dataloader)
    avg_total_loss = running_total_loss / num_batches
    avg_engagement_loss = running_engagement_loss / num_batches
    avg_toxicity_loss = running_toxicity_loss / num_batches

    # Convert probabilities to binary predictions (threshold = 0.5)
    toxicity_preds_bin = [1 if p >= 0.5 else 0 for p in all_toxicity_preds]
    engagement_preds_bin = [1 if p >= 0.5 else 0 for p in all_engagement_preds]

    try:
        tox_roc_auc = roc_auc_score(all_toxicity_labels, all_toxicity_preds)
    except ValueError:
        tox_roc_auc = float('nan') # Handle cases where only one class is present in target

    try:
        eng_roc_auc = roc_auc_score(all_engagement_labels, all_engagement_preds)
    except ValueError:
        eng_roc_auc = float('nan')

    tox_acc = accuracy_score(all_toxicity_labels, toxicity_preds_bin)
    eng_acc = accuracy_score(all_engagement_labels, engagement_preds_bin)

    return {
        "model_name": model_name,
        "Total Loss": avg_total_loss,
        "Eng Loss": avg_engagement_loss,
        "Tox Loss": avg_toxicity_loss,
        "Eng ROC-AUC": eng_roc_auc,
        "Tox ROC-AUC": tox_roc_auc,
        "Eng Acc": eng_acc,
        "Tox Acc": tox_acc
    }


def load_model_checkpoint(model, model_name, device):
    checkpoint_path = f"ModelBinaries/{model_name}/{model_name}_best_model.pth"
    if not os.path.exists(checkpoint_path):
        # Try finding the last checkpoint if best is not there
        checkpoint_path = f"ModelBinaries/{model_name}/{model_name}_last_checkpoint.pth"
        if not os.path.exists(checkpoint_path):
            logger.warning(f"No checkpoint found for {model_name} at {checkpoint_path}")
            return False

    logger.info(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    return True
def save_sample_predictions_md(models, valid_dataset, test_dataset, device, filename="test_predictions.md", num_samples=10):
    logger.info(f"Saving sample predictions to {filename}...")
    
    valid_indices = random.sample(range(len(valid_dataset)), min(num_samples, len(valid_dataset)))
    test_indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# MTL Model Predictions Report\n\n")
        
        # --- Validation Set Output ---
        f.write("## 1. Validation Set (With True Labels)\n")
        f.write("> **Note:** The validation set contains true ground-truth targets for Toxicity and Engagement.\n\n")
        
        for idx in valid_indices:
            df_row = valid_dataset.toxicity_df.iloc[idx]
            comment = df_row['comment_text']
            comment_id = df_row.get('id', idx)
            true_tox = df_row.get('toxic_label', 'N/A')
            true_eng = df_row.get('click_label', 'N/A')
            
            f.write(f"### Sample ID: `{comment_id}`\n")
            comment_quoted = str(comment).replace('\n', '\n> ')
            f.write(f"**Comment:**\n> {comment_quoted}\n\n")
            f.write(f"- **True Toxicity:** {'Toxic' if true_tox == 1 else 'Safe'}\n")
            f.write(f"- **True Engagement:** {'Clicked' if true_eng == 1 else 'Ignored'}\n\n")
            
            f.write("| Model | Toxicity Prob | Engagement Prob |\n")
            f.write("|-------|---------------|-----------------|\n")
            
            inputs = valid_dataset.comment_text_embd[idx]
            inputs_tensor = torch.tensor(inputs).unsqueeze(0).to(device)
            
            for model_name, model in models.items():
                model.eval()
                with torch.no_grad():
                    eng_out, tox_out = model(inputs_tensor)
                    eng_prob = eng_out.item()
                    tox_prob = tox_out.item()
                    
                    tox_str = f"**{tox_prob:.4f}**" if tox_prob > 0.5 else f"{tox_prob:.4f}"
                    eng_str = f"**{eng_prob:.4f}**" if eng_prob > 0.5 else f"{eng_prob:.4f}"
                    
                    f.write(f"| {model_name} | {tox_str} | {eng_str} |\n")
            f.write("\n---\n\n")


        # --- Test Set Output ---
        f.write("## 2. True Test Set (Inference Only)\n")
        f.write("> **Note:** The Kaggle test set does *not* contain true labels. These metrics represent raw inference probabilities.\n\n")
        
        for idx in test_indices:
            df_row = test_dataset.toxicity_df.iloc[idx]
            comment = df_row['comment_text']
            comment_id = df_row.get('id', idx)
            
            f.write(f"### Sample ID: `{comment_id}`\n")
            comment_quoted = str(comment).replace('\n', '\n> ')
            f.write(f"**Comment:**\n> {comment_quoted}\n\n")
            
            f.write("| Model | Toxicity Prob | Engagement Prob |\n")
            f.write("|-------|---------------|-----------------|\n")
            
            inputs = test_dataset.comment_text_embd[idx]
            inputs_tensor = torch.tensor(inputs).unsqueeze(0).to(device)
            
            for model_name, model in models.items():
                model.eval()
                with torch.no_grad():
                    eng_out, tox_out = model(inputs_tensor)
                    eng_prob = eng_out.item()
                    tox_prob = tox_out.item()
                    
                    tox_str = f"**{tox_prob:.4f}**" if tox_prob > 0.5 else f"{tox_prob:.4f}"
                    eng_str = f"**{eng_prob:.4f}**" if eng_prob > 0.5 else f"{eng_prob:.4f}"
                    
                    f.write(f"| {model_name} | {tox_str} | {eng_str} |\n")
            f.write("\n---\n\n")
            
    logger.info("Sample predictions saved successfully.")


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # We use validation split from train dataset because test.csv does not contain target labels
    logger.info("Loading validation dataset from train split...")
    _, toxicity_valid_df = load_dataset(split="train")
    
    # Passing train=True to get targets from dataset
    toxicity_valid_dataset = ToxicityDataset(toxicity_valid_df, embedding_model, train=True)
    test_dataloader = DataLoader(toxicity_valid_dataset, batch_size=32, shuffle=False)
    
    criterion = [nn.BCELoss(), nn.BCELoss()]
    results = []
    loaded_models = {}

    # 1. Evaluate SharedBottomMLT
    sb_model = SharedBottomMLT().to(device)
    if load_model_checkpoint(sb_model, "SharedBottomModel", device):
        res = evaluate_model(sb_model, test_dataloader, criterion, device, "SharedBottomModel")
        results.append(res)
        loaded_models["SharedBottomModel"] = sb_model
        
    # 2. Evaluate BasicMoE
    moe_model = BasicMoE(input_dim=384, output_dim=128, num_experts=3).to(device)
    if load_model_checkpoint(moe_model, "BasicMoE", device):
        res = evaluate_model(moe_model, test_dataloader, criterion, device, "BasicMoE")
        results.append(res)
        loaded_models["BasicMoE"] = moe_model

    # 3. Evaluate BasicMMoE
    mmoe_model = BasicMMoE(input_dim=384, output_dim=128, num_experts=3).to(device)
    if load_model_checkpoint(mmoe_model, "BasicMMoE", device):
        res = evaluate_model(mmoe_model, test_dataloader, criterion, device, "BasicMMoE")
        results.append(res)
        loaded_models["BasicMMoE"] = mmoe_model

    if not results:
        logger.warning("No models were successfully loaded and evaluated.")
        return

    # Print nicely formatted results
    logger.info("="*85)
    logger.info(f"{'Model Name':<20} | {'Total Loss':<10} | {'Eng Loss':<8} | {'Tox Loss':<8} | {'Eng AUC':<7} | {'Tox AUC':<7} | {'Eng Acc':<7} | {'Tox Acc':<7}")
    logger.info("-" * 85)
    for res in results:
        logger.info(f"{res['model_name']:<20} | {res['Total Loss']:<10.4f} | {res['Eng Loss']:<8.4f} | {res['Tox Loss']:<8.4f} | {res['Eng ROC-AUC']:<7.4f} | {res['Tox ROC-AUC']:<7.4f} | {res['Eng Acc']:<7.4f} | {res['Tox Acc']:<7.4f}")
    logger.info("="*85)

    if loaded_models:
        # Load the test dataset (which doesn't have targets) to run pure predictions
        logger.info("Loading true test dataset for sample inference...")
        toxicity_test_df = load_dataset(split="test")
        
        # We pass train=False so dataset loader doesn't try looking for missing targets
        toxicity_test_dataset = ToxicityDataset(toxicity_test_df, embedding_model, train=False)
        
        
        save_sample_predictions_md(loaded_models, toxicity_valid_dataset, toxicity_test_dataset, device, filename="test_predictions.md", num_samples=10)


if __name__ == "__main__":
    main()
