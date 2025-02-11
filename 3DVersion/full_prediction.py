import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from Data import BasicDataset
from model import Model
from predict import add_mirrored_slices, predict_stack, remove_mirrored_slices
from skimage.segmentation import find_boundaries, mark_boundaries

device_id = 1
path_output = "<yourPath>/NucleusPanVision/predictions3D"
path_output_inspect = "<yourPath>/NucleusPanVision/predictions3D_inspect"

prediction_window_size = (32, 512, 512)
prediction_stride = (16, 512, 512)


def clean_file_name(name):
    name = name.replace(".ims Resolution Level 1", "")
    name = name.replace(".ims_Resolution_Level_1", "")
    name = name.replace("_TOM20647_Mitotracker_NHSester488", "")
    return name


def save_quick_view(nhs, pred, img_name):
    nrow, ncol = 8, 4
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
    ax = ax.ravel()

    indices = np.linspace(0, nhs.shape[0] - 1, len(ax)).astype("uint8")

    pred[:, 0, 0] = 2
    for i, idx in enumerate(indices):
        ax[i].imshow(nhs[idx], cmap="gray")
        boundaries = find_boundaries(pred[idx], mode="inner")
        ax[i].imshow(pred[idx], alpha=0.2)
        ax[i].imshow(boundaries, alpha=0.9 * (boundaries > 0), cmap="Reds")
        # ax[i].imshow(mark_boundaries(nhs[idx], pred[idx], ))
        ax[i].axis("off")
        ax[i].set_title(f"z={idx}")
    if not os.path.exists(os.path.dirname(img_name)):
        os.makedirs(os.path.dirname(img_name), exist_ok=True)
    plt.tight_layout()
    plt.savefig(img_name)
    plt.close()


files = [
    "Hela_Tom20_MitoOrange_2022-07-12_1.ims_Resolution_Level_1.tif",
]


model = Model()
model = model.load_from_checkpoint(
    "<yourPath>/NucleusPanVision/3DVersion/result/Run1/checkpoints/Run1.ckpt"
)
model = model.to(torch.device(f"cuda:{device_id}"))


failed = []
for file in files:
    try:
        print(f'{time.strftime("%H:%M:%S")} Predicting: {file}')
        output_file = os.path.join(path_output, clean_file_name(file)).replace(
            ".tif", "_pred.tif"
        )
        if os.path.exists(output_file):
            continue

        data = BasicDataset(
            imgs_path_list=[file],
            training_size=(
                32,
                512,
                512,
            ),  # Irrelevant... only the prediction window at the top is used!
            data_stride=(16, 512, 512),
            extract_channel=2,
            mode="test",
            xy_factor=4,
            load_target=False,
        )
        cur_img = data.img_list[0]
        cur_img = add_mirrored_slices(cur_img, prediction_window_size[0] // 2)
        prediction = predict_stack(
            cur_img, model, img_window=prediction_window_size, stride=prediction_stride
        )
        prediction = remove_mirrored_slices(prediction, prediction_window_size[0] // 2)
        cur_img = remove_mirrored_slices(cur_img, prediction_window_size[0] // 2)

        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        tifffile.imwrite(
            output_file,
            prediction.astype("uint8"),
            imagej=True,
            metadata={"axes": "ZYX"},
            compression="zlib",
        )

        save_quick_view(
            cur_img.cpu().numpy(),
            prediction,
            os.path.join(
                path_output_inspect, clean_file_name(file).replace(".tif", ".png")
            ),
        )

    except Exception as e:
        print(f"Error in {file}: {e}")
        failed.append((file, e))
        continue

print("Failed: ", failed)
print("Errors: ", [f[0] for f in failed])
