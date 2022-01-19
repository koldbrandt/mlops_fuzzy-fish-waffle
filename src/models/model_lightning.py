import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import pytorch_lightning as pl

# wandb.init(project="mlops-project", entity="fuzzy-fish-waffle")

class LightningModel(pl.LightningModule):
    def __init__(self, num_classes: int) -> None:
        """Builds a feedforward network with arbitrary hidden layers.

        Arguments
        ---------
        output_size: number of classes
        """
        super().__init__()
        # Input to a hidden layer
        self.num_classes = num_classes

        self.layers = nn.Sequential(
            # 1st Convolution
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # 2nd Convolution
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # 3rd Convolution
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # MLP Layer
            nn.Flatten(),
            nn.Linear(16384, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        # Image has to have shape 64 x 64
        if not x.shape[2:4] == torch.Size([64, 64]):
            raise ValueError("Expected each image to have shape 64 x 64")

        logits = self.layers(x)
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        images, labels = batch
        x_hat = self.layers(images)
        loss = F.cross_entropy(x_hat, labels)
        self.log("train_loss", loss)
        self.logger.experiment.log({"logits": wandb.Histogram(x_hat.detach().numpy())})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=0.01,
            momentum=0.9,
        )
        return optimizer

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        y_hat = self.layers(images)
        val_loss = F.cross_entropy(y_hat, labels)
        self.log("val_loss", val_loss)
        return val_loss