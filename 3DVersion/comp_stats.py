import tifffile
import os
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
import re


def extract_digits(s):
    # Get the file name from s
    s = s.split("/")[-1]
    # Define the regex pattern to match the digits
    pattern = r"_(\d{1,2})_"

    # Search for the pattern in the string
    match = re.search(pattern, s)

    # If match is found, return the digits, otherwise return 0
    if match:
        return match.group(1)
    else:
        return "0"


def process_prediction(file):
    pred = tifffile.imread(file)
    print(time.strftime("%H:%M:%S"), "Processing: ", file)

    values, counts = np.unique(pred, return_counts=True)

    val2label = {0: "Background", 1: "Cytoplasm", 2: "Nucleus"}
    values = [val2label[val] for val in values]

    if "2nd" in file:
        batch = 2
    elif "3rd" in file:
        batch = 3
    else:
        batch = 1

    if "DMSO" in file:
        drug = "DMSO"
    elif "Oligo" in file:
        drug = "Oligomycin"
    elif "Antim" in file:
        drug = "Antimycin"
    elif "HBSS" in file:
        drug = "HBSS"
    elif "CCCP" in file:
        drug = "CCCP"
    elif "Control" in file:
        drug = "Control"
    else:
        drug = "Unkown"

    cell = extract_digits(file)
    source = file.split("/")[-1].replace("_pred.tif", ".tif") + f"_b{batch}"

    return {
        **{"file": file},
        **{label: value for label, value in zip(values, counts.tolist())},
        **{"batch": batch},
        **{"cell": cell},
        **{"drug": drug},
        **{"source": source},
        **{dim: length for dim, length in zip(["z", "y", "x"], pred.shape)},
    }


if __name__ == "__main__":
    path_output = "<yourPath>/NucleusPanVision/predictions3D"

    # Walk directory and find all tif files
    tif_files = []
    for root, dirs, files in os.walk(path_output):
        for file in files:
            if file.endswith(".tif") and not file.startswith("."):
                tif_files.append(os.path.join(root, file))

    pool = mp.Pool(8)
    statistics = pool.map(process_prediction, tif_files)

    results = pd.DataFrame(statistics)
    results.to_csv(os.path.join(path_output, "statistics.csv"), index=False)
