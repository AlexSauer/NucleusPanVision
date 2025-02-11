import argparse

import pytorch_lightning as pl
from Data import CellData
from model import Model
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)

path_result = "<yourPath>/NucleusPanVision/3DVersion/result"
pl.seed_everything(123456)
args = argparse.ArgumentParser()
args.add_argument("--run_name", type=str, default="Test")
args.add_argument("--lr", type=float, default=1e-4)
args.add_argument("--weight_decay", type=float, default=1e-4)
args.add_argument("--device", type=int, default=2)
args.add_argument("--batch_size", type=int, default=2)
args = args.parse_args()


files = [
    "Hela_Tom20_MitoOrange_2022-07-12_1.ims_Resolution_Level_1.tif",
]


data = CellData(
    train_files=files,
    test_files=files,
    training_size=(16, 512, 512),
    data_stride=(16, 512, 512),
    extract_channel=2,
    batch_size=args.batch_size,
    xy_factor=4,
)

model = Model(lr=args.lr, weight_decay=args.weight_decay)
logger = pl.loggers.TensorBoardLogger(path_result, name=args.run_name)
early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)
checkpoint_callback = ModelCheckpoint(every_n_epochs=5, save_top_k=-1)

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=[args.device],
    logger=logger,
    callbacks=[
        early_stop,
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
    ],
)

trainer.fit(model, data)
