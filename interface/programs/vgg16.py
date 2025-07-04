import torch
import torch.nn as nn
import lightning.pytorch as pl

from torchvision import models

class VGG16TransferLearning(pl.LightningModule):
    def __init__(self, num_classes, lr_phase1=1e-4, lr_phase2=1e-5, fine_tune=False):
        super().__init__()
        self.save_hyperparameters()

        self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.model.avgpool =  nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=256, out_features=64, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=64, out_features=num_classes, bias=True)
        )

        for param in self.model.features.parameters():
            param.requires_grad = False

        if fine_tune:
            for param in self.model.features[-4:].parameters():
                param.requires_grad = True

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_acc", acc, on_epoch=True)

    def configure_optimizers(self):
        lr = self.hparams.lr_phase2 if self.hparams.fine_tune else self.hparams.lr_phase1
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        # [p for p in self.parameters() if p.requires_grad]
        return optimizer