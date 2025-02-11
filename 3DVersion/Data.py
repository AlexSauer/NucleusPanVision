import os
from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import skimage
import torch
from einops import rearrange
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset

PATH_ORG = "<yourPath>/projects/PanVision/data/FullStacks/Originals"
PATH_PRED = "<yourPath>/projects/PanVision/data/FullStacks/Annotation/NucleusBackground"


def clean_file_name(name):
    name = name.replace(".ims Resolution Level 1", "")
    name = name.replace(".ims_Resolution_Level_1", "")
    name = name.replace("_TOM20647_Mitotracker_NHSester488", "")
    return name


class BasicDataset(Dataset):
    def __init__(
        self,
        imgs_path_list: List[str],
        #  targets_path_list: Optional[List[str]] = None,
        training_size: Tuple[int] = (32, 256, 256),
        data_stride: Tuple[int] = (16, 128, 128),
        mode: str = "train",
        extract_channel: Optional[int] = None,
        xy_factor: int = 4,
        load_target: bool = True,
    ):
        self.imgs_path_list = [os.path.join(PATH_ORG, file) for file in imgs_path_list]
        if load_target:
            self.targets_path_list = [
                os.path.join(PATH_PRED, clean_file_name(file))
                for file in imgs_path_list
            ]
        else:
            self.targets_path_list = [""] * len(imgs_path_list)
        self.training_size = training_size
        self.data_stride = data_stride
        self.mode = mode
        assert self.mode in ["train", "val", "test"], f"Unvalid mode parameter: {mode}"

        self.img_list = []
        self.target_list = []

        self.ids = []  # List of extracted patches
        self.nhs_quantiles = [None] * len(imgs_path_list)
        self.nhs_mins = [None] * len(imgs_path_list)

        for file_id in range(0, len(self.imgs_path_list)):
            # Read files
            imgs_path = self.imgs_path_list[file_id]
            target_path = self.targets_path_list[file_id]
            cur_image = skimage.io.imread(imgs_path)

            if cur_image.ndim == 3:
                cur_image = cur_image[..., None]

            # When we load in the full data, we often want to extract a specific channel (mostly NHS, i.e. channel 2)
            if extract_channel is not None:
                channel_dim = np.argmin(
                    cur_image.shape
                )  # We assume that the channel dimension is the smallest
                cur_image = np.take(
                    cur_image, indices=extract_channel, axis=channel_dim
                )

            if target_path != "":
                # Load the target if its present (won't be if we use the BasicDataset for inference)
                self.target_set = skimage.io.imread(target_path).astype(
                    "uint8"
                )  # /255 # Empty masks get loaded as uint16 for some reason
            else:
                self.target_set = np.zeros(cur_image.shape)

            # Scale the image
            cur_image = resize(
                cur_image,
                (
                    cur_image.shape[0],
                    cur_image.shape[1] / xy_factor,
                    cur_image.shape[2] / xy_factor,
                ),
                anti_aliasing=True,
                preserve_range=True,
            )
            if load_target:
                self.target_set = resize(self.target_set, cur_image.shape, order=0)

            # Preprocess the data and normalise the data -> 0-1
            # For the training mode, we precompute the cut-off quantiles (acts as augmentation)
            # Otherwise, if no threshold config file is given which specifies a specific max value,
            # we compute the 99% quantile and set this to 1
            # Specifically for the 2nd batch we need to give hand-picked thresholds as the data looks very different
            if self.mode == "train":
                self.nhs_quantiles[file_id] = np.quantile(
                    cur_image, q=[0.985, 0.9875, 0.99, 0.9925, 0.995, 0.9975]
                )
                self.nhs_mins[file_id] = np.min(cur_image)
            else:
                max_nhs, min_nhs = np.quantile(cur_image, q=0.99), np.min(cur_image)
                cur_image = np.clip(cur_image, min_nhs, max_nhs)
                cur_image = (cur_image - min_nhs) / (max_nhs - min_nhs)

            self.img_list.append(
                torch.tensor(cur_image.astype(np.float32), dtype=torch.float)
            )
            self.target_list.append(torch.tensor(self.target_set, dtype=torch.float))

            end_point = cur_image.shape
            start_point = (0, 0, 0)
            # Find the positions of each extracted patch
            for i in range(
                start_point[0],
                end_point[0] - self.training_size[0] + 1,
                self.data_stride[0],
            ):
                for x in range(
                    start_point[1],
                    end_point[1] - self.training_size[1] + 1,
                    self.data_stride[1],
                ):
                    for y in range(
                        start_point[2],
                        end_point[2] - self.training_size[2] + 1,
                        self.data_stride[2],
                    ):
                        self.ids.append(
                            [
                                file_id,
                                [i, i + training_size[0]],
                                [x, x + training_size[1]],
                                [y, y + training_size[2]],
                            ]
                        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        # Extract the patch coordinates
        file_id, z_range, x_range, y_range = self.ids[i]

        # Extract image sets
        img = self.img_list[file_id][
            z_range[0] : z_range[1], x_range[0] : x_range[1], y_range[0] : y_range[1]
        ].clone()
        target = self.target_list[file_id][
            z_range[0] : z_range[1], x_range[0] : x_range[1], y_range[0] : y_range[1]
        ].clone()

        if self.mode == "train":
            # Flipping - data augmentation
            flip_data = np.random.rand(3) > 0.5
            for dim in range(3):
                if flip_data[dim]:
                    img = torch.flip(img, (dim,))
                    target = torch.flip(target, (dim,))

            # Thresholding augmentation
            # Select random quantile
            quantile_id = np.random.randint(0, len(self.nhs_quantiles[file_id]))
            quantile = self.nhs_quantiles[file_id][quantile_id]
            cur_min = self.nhs_mins[file_id]

            img = (img.clip(cur_min, quantile) - cur_min) / (quantile - cur_min)

        img = BasicDataset.rearrange_shape(img)
        target = BasicDataset.rearrange_shape_target(target)

        return {"image": img, "target": target}

    @staticmethod
    def rearrange_shape(img_trans):
        if len(img_trans.shape) == 3:
            img_trans = img_trans[..., None]
        # HWC to CHW
        img_trans = rearrange(img_trans, "Z X Y C-> C Z X Y")

        return img_trans

    @staticmethod
    def add_gaussian_noise(img, std: float = 0.0025):
        noise = std * torch.randn(*img.shape)
        return img + noise

    @staticmethod
    def rearrange_shape_target(target):
        if len(target.shape) == 3:
            target = target[..., None]

        # HWC to CHW
        target = rearrange(target, "Z X Y C-> C Z X Y")
        return target


class CellData(pl.LightningDataModule):
    def __init__(
        self,
        train_files,
        test_files,
        batch_size,
        training_size: Tuple[int] = (32, 256, 256),
        data_stride: Tuple[int] = (16, 128, 128),
        extract_channel: Optional[int] = None,
        xy_factor: int = 4,
    ):
        super().__init__()
        self.train_files = train_files
        self.test_files = test_files
        self.batch_size = batch_size
        self.training_size = training_size
        self.data_stride = data_stride
        self.extract_channel = extract_channel
        self.xy_factor = xy_factor

    def setup(self, stage=None):
        self.train_data = BasicDataset(
            self.train_files,
            training_size=self.training_size,
            data_stride=self.data_stride,
            mode="train",
            extract_channel=self.extract_channel,
            xy_factor=self.xy_factor,
        )
        self.test_data = BasicDataset(
            self.test_files,
            training_size=self.training_size,
            data_stride=self.data_stride,
            mode="test",
            extract_channel=self.extract_channel,
            xy_factor=self.xy_factor,
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.test_data, self.batch_size)

    # def test_dataloader(self):
    #     return DataLoader(ImgDataSet(self.test_img, self.test_annotation, stage='test'), self.batch_size)
