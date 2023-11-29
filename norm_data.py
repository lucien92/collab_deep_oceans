import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path

path = Path('./npy/plankton_med-npy/')

meds, perc1, perc99 = np.load("stats/stats.npy")

print("Starting processing files...")

for (i, f) in enumerate(tqdm(path.iterdir(), desc="Processing")):
    print(f"Loading file: {f.name}")
    item = np.load(f)

    # # Fill NaNs with median values (tile median or, if empty, dataset median)
    # all_nans = np.isnan(item).all(axis=(0,1))
    # some_nans = np.logical_and(~all_nans, np.isnan(item).any(axis=(0,1)))
    # fill_values = all_nans * meds + some_nans * np.nan_to_num(np.nanmedian(item, axis = [0,1]), nan=0)
    # filled = np.nan_to_num(item, nan=fill_values)

    # Actually we want to fill NaN with 0s : 
    filled = np.nan_to_num(item, nan=0)

    print(f"NaNs handled for file: {f.name}")

    # Normalize
    normed = (filled - perc1) / (perc99 - perc1)
    print(f"Normalization done for file: {f.name}")

    # Remove extreme values
    final = np.clip(normed, 0, 1)
    print(f"Extreme values clipped for file: {f.name}")

    # Save
    # np.save('npy/plankton_med-npy-norm/' + f.name, final)
    np.save('npy/plankton_med-npy-norm-zero-fill/' + f.name, final)
    print(f"Processed file saved: {f.name}")

print("All files processed.")
