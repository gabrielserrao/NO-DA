#%%
import sys
sys.path.append("../..")
from darts.engines import redirect_darts_output
from model_co2 import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.mplot3d import Axes3D
from utilsDARTS import ModelOut, InitializeDataVars, StoreSimValues,create_wells_dataset
import os
# %%

