import pandas as pd
import numpy as np
import geoenrich
from geoenrich import dataloader, enrichment, satellite, exports
import pathlib

# For each variables, we want to plot the distribution of NaN values : 
# Given one variable foreach observation, we will get the ratio of NaN and then aggregate all those ratios

# iterate over the npy files (in /npy/plankton_med-npy)

folder = pathlib.Path('npy/plankton_med-npy')
files = list(folder.glob('*.npy'))  # Convert to list to reuse

nan_ratios = []
n_files = len(files)

# enumerate : 
for i, file in enumerate(files):
    raster = np.load(file)
    # Get the number of NaN:
    nan_ratio = np.isnan(raster).sum(axis = (0, 1))/(raster.shape[0]*raster.shape[1])
    nan_ratios.append(nan_ratio)
    print(file)
    print(nan_ratio)
    print('\n')
    print("Progress : ", i, "/", n_files, " (", round(i/n_files*100, 2), "%)")
    print('\n')


# Save
nan_ratios = np.array(nan_ratios)
np.save('stats/nan_ratios.npy', nan_ratios)