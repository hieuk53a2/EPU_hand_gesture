import torch
import torch.nn as nn
import torchvision.models as models

class MultiInputResNet(nn.Module):
    def __init__(self, num_classes, sensor_input_dim=5, pretrained=True):
        super(MultiInputResNet, self).__init__()

        # Load ResNet50 without the final classification layer
        resnet = models.resnet50(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
        self.image_feature_dim = resnet.fc.in_features  # Feature output size

        # Define FCN for sensor input
        self.sensor_fc = nn.Sequential(
            nn.Linear(sensor_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # Softmax layer
        self.softmax = nn.Softmax(dim=1)  

        # Final classifier after concatenation
        self.classifier = nn.Sequential(
            nn.Linear(self.image_feature_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, sensor_data):
        # Extract features from the image using ResNet
        image_features = self.feature_extractor(image)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten

        # Apply softmax to image features
        image_features = self.softmax(image_features)

        # Process sensor input using FCN
        sensor_features = self.sensor_fc(sensor_data)
        sensor_features = self.softmax(sensor_features)  # Apply softmax to sensor features

        # Concatenate after softmax
        combined_features = torch.cat((image_features, sensor_features), dim=1)
        print("#########", image_features, sensor_features, combined_features)
        # Final classification
        output = self.classifier(combined_features)
        return output
