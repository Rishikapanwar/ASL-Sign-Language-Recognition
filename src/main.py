import torch
import wandb
from torch.utils.data import DataLoader, random_split
from torch import nn, optim

from custom_dataset import ASLDataset, collate_fn
from lstm import ASLLSTM, train
from feature_extraction import MediaPipe
from utils import TRAIN_DIR


def main(wandb_logging):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    dataset = ASLDataset(
        data_dir=TRAIN_DIR,
        extractor=MediaPipe(),
        transform=None,
        # limit_count=10     # Small count per class
    )
    val_size = int(0.2*len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    num_classes = len(dataset.class_to_idx.keys())

    model = ASLLSTM(input_size=3, hidden_size=128, num_layers=2, num_classes=num_classes)
    model.load_state_dict(torch.load(r'C:\Users\ypanw\PycharmProjects\PythonProject\checkpoints\asl_lstm_final.pth'))
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("Starting training...")
    train(model, criterion, optimizer, train_loader, val_loader=val_loader, epochs=1, print_freq=1, device=device,wandb_logging=wandb_logging)

if __name__ == '__main__':
    wandb_logging=True
    if wandb_logging:
        wandb.init(entity='personal_project_rishika',
                   project='slr',
                   name='final_run'
                )
    main(wandb_logging)
