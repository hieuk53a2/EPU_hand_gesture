import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import os
from PIL import Image

class MultiInputDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # Scan all class directories
        for label, class_dir in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):  # Ensure it's a folder
                for file in sorted(os.listdir(class_path)):
                    if file.endswith(".png"):
                        file_name = file.split("_")[0]+".csv"
                        img_path = os.path.join(class_path, file)
                        csv_path = os.path.join(class_path, file_name)
                        if os.path.exists(csv_path):  # Ensure CSV exists
                            self.data.append((img_path, csv_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, csv_path, label = self.data[idx]
        # print(img_path, csv_path, label)
        # Load Image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load Sensor Data
        sensor_data = pd.read_csv(csv_path, header=None, sep='\s+').values.flatten().astype("float32")

        
        sensor_data = torch.tensor(sensor_data)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, sensor_data, label

def get_dataloaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MultiInputDataset(os.path.join(data_dir, 'train'), transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = MultiInputDataset(os.path.join(data_dir, 'val'), transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # get number of classes
    num_classes = len(set([x[2] for x in train_dataset.data]))
    return train_dataloader, val_dataloader, num_classes   
