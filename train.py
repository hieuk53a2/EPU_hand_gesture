import torch
import torch.optim as optim
import torch.nn as nn
from models.multi_input_resnet import MultiInputResNet
from datasets.dataloader import get_dataloaders
from utils.train_utils import train, validate
import config

def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    train_loader, val_loader, num_classes = get_dataloaders(config.DATA_DIR, config.BATCH_SIZE)
    print(f"Number of classes: {num_classes}")
    # Load model
    model = MultiInputResNet(num_classes).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    # Training loop
    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{config.EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Save trained model
    torch.save(model.state_dict(), "resnet50_custom.pth")


if __name__ == "__main__":
    main()