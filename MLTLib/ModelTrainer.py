import torch

class ModelTrainer:
    def __init__(self, model, train_dataloader, valid_dataloader, criterion, optimizer, device, model_name):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name

    def train_one_epoch(self, print_every=25):
        running_loss = 0.
        last_loss = 0.
        
        for i, data in enumerate(self.train_dataloader):
            inputs, toxicity_label, engagement_label = data
            inputs, toxicity_label, engagement_label = inputs.to(self.device), toxicity_label.to(self.device), engagement_label.to(self.device)
            
            self.optimizer.zero_grad()
            
            eng_out, tox_out = self.model(inputs)
        
            engagement_loss = self.criterion[0](eng_out.view(-1), engagement_label.float())
            toxicity_loss = self.criterion[1](tox_out.view(-1), toxicity_label.float())
        
            total_loss = engagement_loss + toxicity_loss
            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item()
            
            if (i+1) % print_every == 0:
                last_loss = running_loss/print_every
                print(f"Batch {i+1}: Loss = {last_loss:.4f}")

        return last_loss

    def train(self, num_epochs=5):
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            train_loss = self.train_one_epoch()
            valid_loss = self.evaluate()
            
            print(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(self.model.state_dict(), f"{self.model_name}_best_model.pth")
                print("Saved best model")

    def evaluate(self):
        self.model.eval()
        running_loss = 0.
        
        for i, data in enumerate(self.valid_dataloader):
            inputs, toxicity_label, engagement_label = data
            inputs, toxicity_label, engagement_label = inputs.to(self.device), toxicity_label.to(self.device), engagement_label.to(self.device)
            
            eng_out, tox_out = self.model(inputs)
        
            engagement_loss = self.criterion[0](eng_out.view(-1), engagement_label.float())
            toxicity_loss = self.criterion[1](tox_out.view(-1), toxicity_label.float())
        
            total_loss = engagement_loss + toxicity_loss
            running_loss += total_loss.item()
        
        return running_loss/len(self.valid_dataloader)