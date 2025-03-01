import torch

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, sensor_values, labels in train_loader:
        images, sensor_values, labels = images.to(device), sensor_values.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, sensor_values)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(train_loader), 100 * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, sensor_values, labels in val_loader:
            images, sensor_values, labels = images.to(device), sensor_values.to(device), labels.to(device)
            outputs = model(images, sensor_values)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(val_loader), 100 * correct / total
