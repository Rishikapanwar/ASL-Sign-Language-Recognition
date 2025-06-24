import torch
import os
import torch.nn as nn
from tqdm import tqdm
from configs import checkpoints_dir
import wandb

class ASLLSTM(nn.Module):

    def __init__(self, input_size=3, hidden_size=128, num_layers=2, num_classes=29):
        super(ASLLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        out = self.classifier(hn[-1])
        return out



def train(model, criterion, optimizer, train_loader, val_loader, epochs, print_freq, device,wandb_logging):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")
        for step, batch in loop:

            inputs, labels = batch
            if inputs is None or labels is None:
                print(f"[INFO] Skipping empty batch at step {step}")
                continue
            inputs = inputs.to(device).view(inputs.size(0), 21, 3)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        print(f"[Train] Epoch {epoch + 1}, Avg Loss: {avg_train_loss:.4f}")
        if wandb_logging:
            wandb.log({"Training Loss": avg_train_loss})

        if val_loader is not None:
            print('now in eval mode')
            model.eval()
            total, correct = 0, 0
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    if val_inputs is None or val_labels is None:
                        print(f"[INFO] Skipping empty batch at step {step}")
                        continue
                    val_inputs = val_inputs.to(device).view(val_inputs.size(0), 21, 3)
                    val_labels = val_labels.to(device)

                    val_outputs = model(val_inputs)
                    _, predicted = torch.max(val_outputs, dim=1)

                    total += val_labels.size(0)
                    correct += (predicted == val_labels).sum().item()

            accuracy = 100 * correct / total
            if wandb_logging:
                wandb.log({"Epoch: ":epoch+1})
                wandb.log({"Validation Accuracy": accuracy})
            print(f"[Val] Epoch {epoch + 1}, Accuracy: {accuracy:.2f}%")

            # Optional: show predictions for first 5 validation samples
            print("Sample Predictions vs Labels:")
            print("Pred:", predicted[:5].tolist())
            print("True :", val_labels[:5].tolist())
            if wandb_logging:
                wandb.log({"Pred:": predicted[:5].tolist()})
                wandb.log({"True :": val_labels[:5].tolist()})

    torch.save(model.state_dict(), os.path.join(checkpoints_dir, f"asl_lstm.pth"))
    print(f"Model saved: asl_lstm.pth")
