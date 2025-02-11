import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from UNetFactory import UNet, SegmentatorNetwork

class Model(pl.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.embedding_network = UNet(encoder_channels=[1, 64, 128, 256, 512],
                                           decoder_channels=[512, 256, 128, 64, 32, 1],
                                           residual=True,
                                           type='3D')
        self.segmentation_head = SegmentatorNetwork(3, in_classes=32)
        self.loss = torch.nn.CrossEntropyLoss()
        self.weight_decay = weight_decay
        self.lr = lr

    def forward(self, x):
        return self.segmentation_head(self.embedding_network(x))    

    def comp_loss(self, batch):
        x, y = batch['image'], batch['target'].squeeze(1).long()
        pred = self.segmentation_head(self.embedding_network(x))

        loss = self.loss(pred, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.comp_loss(batch)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.comp_loss(batch)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': StepLR(optimizer, step_size=10, gamma=0.1),  # Stepwise scheduler after 5 epochs
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]
    
