import torch
import logging
from tqdm import tqdm
import os

class ModelTrainer:
    def __init__(self, model, train_dataloader, valid_dataloader, criterion, optimizer, device, model_name, patience=3):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.patience = patience  # For early stopping
        
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger(self.model_name)
        self.logger.setLevel(logging.INFO)
        # Avoid duplicate logs if logger is already configured
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
            # Optionally, add a file handler to save logs to a file
            fh = logging.FileHandler(f"{self.model_name}_training.log")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.
        total_loss, engagement_loss, toxicity_loss = 0, 0, 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
        
        for i, data in enumerate(progress_bar):
            inputs, toxicity_label, engagement_label = data
            inputs = inputs.to(self.device)
            toxicity_label = toxicity_label.to(self.device)
            engagement_label = engagement_label.to(self.device)
            
            self.optimizer.zero_grad()
            
            eng_out, tox_out = self.model(inputs)
        
            engagement_loss = self.criterion[0](eng_out.view(-1), engagement_label.float())
            toxicity_loss = self.criterion[1](tox_out.view(-1), toxicity_label.float())
        
            total_loss = engagement_loss + toxicity_loss
            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item()
            
            # Update tqdm progress bar description with current loss
            progress_bar.set_postfix({'loss': running_loss / (i + 1)})

        return running_loss / len(self.train_dataloader)

    def train(self, num_epochs=5, resume_checkpoint=None):
        self.logger.info("-" * 50)
        self.logger.info(f"Started Training: {self.model_name}")
        self.logger.info("-" * 50)
        
        start_epoch = 0
        best_loss = float('inf')
        epochs_without_improvement = 0
        
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            self.logger.info(f"Loading checkpoint '{resume_checkpoint}'")
            checkpoint = torch.load(resume_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            self.logger.info(f"Resumed from epoch {start_epoch} with best validation loss: {best_loss:.4f}")

        try:
            for epoch in range(start_epoch, num_epochs):
                train_loss = self.train_one_epoch(epoch + 1)
                valid_loss = self.evaluate(epoch + 1)
                
                self.logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
                
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    epochs_without_improvement = 0
                    self.save_checkpoint(epoch, best_loss, is_best=True)
                else:
                    epochs_without_improvement += 1
                    self.logger.info(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")
                    
                    if epochs_without_improvement >= self.patience:
                        self.logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                        break
                        
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user! Saving latest state...")
            # Using current valid_loss if available, else best_loss
            save_loss = valid_loss if 'valid_loss' in locals() else best_loss
            self.save_checkpoint(epoch, save_loss, is_best=False, filename=f"{self.model_name}_interrupted.pth")
            
        self.logger.info("Training completed.")

    def evaluate(self, epoch=None):
        self.model.eval()
        running_loss = 0.
        
        desc = f"Epoch {epoch} [Valid]" if epoch else "Evaluation"
        progress_bar = tqdm(self.valid_dataloader, desc=desc, leave=False)
        
        with torch.no_grad(): 
            for i, data in enumerate(progress_bar):
                inputs, toxicity_label, engagement_label = data
                inputs = inputs.to(self.device)
                toxicity_label = toxicity_label.to(self.device)
                engagement_label = engagement_label.to(self.device)
                
                eng_out, tox_out = self.model(inputs)
            
                engagement_loss = self.criterion[0](eng_out.view(-1), engagement_label.float())
                toxicity_loss = self.criterion[1](tox_out.view(-1), toxicity_label.float())
            
                total_loss = engagement_loss + toxicity_loss
                running_loss += total_loss.item()
                
                progress_bar.set_postfix({'loss': running_loss / (i + 1)})
            
        return running_loss / len(self.valid_dataloader)

    def save_checkpoint(self, epoch, best_loss, is_best=False, filename=None):
        ## save to ModelBinaries folder under the folder of the name of the model
        if not os.path.exists(f"ModelBinaries/{self.model_name}"):
            os.makedirs(f"ModelBinaries/{self.model_name}")
            
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': best_loss,
        }
        
        if filename is None:
            filename = f"ModelBinaries/{self.model_name}/{self.model_name}_last_checkpoint.pth"
        
        torch.save(state, filename)
        if is_best:
            best_filename = f"ModelBinaries/{self.model_name}/{self.model_name}_best_model.pth"
            torch.save(state, best_filename)
            self.logger.info(f"Saved new best model with validation loss: {best_loss:.4f}")
