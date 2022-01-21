import torch
import torch.nn as nn

# Architecture
class Network(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.layers = torch.nn.Sequential(
            # 1st Convolution
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # 2nd Convolution
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # 3rd Convolution
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # MLP Layer
            nn.Flatten(),
            nn.Linear(16384, 128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if not x.shape[2:4] == torch.Size([64, 64]):
            raise ValueError("Expected each image to have shape 64 x 64")

        logits = self.layers(x)
        return logits


def validation(model, testloader, criterion):
    """Model validation"""
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        # Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = labels.data == ps.max(1)[1]
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy
