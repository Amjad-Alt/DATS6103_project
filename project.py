#%%[markdown]
import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# %%
data = pd.read_csv("restaurant-1-orders.csv")

# %%
data.shape
data.info()
data.describe()
# %%
data["Total Price"] = data["Product Price"] * data["Quantity"]
data
# %%
