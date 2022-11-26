#%%[markdown]
import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# %%
data = pd.read_csv("restaurant-1-orders.csv")

#%%
data.isna().mean()
# %%
data.shape
data.info()
data.describe()
# %%
data["Total Price"] = data["Product Price"] * data["Quantity"]
data
# %%
item_freq = data.groupby('Item Name').agg({'Quantity': 'sum'})
item_freq = item_freq.sort_values(by=['Quantity'])
top_20 = item_freq.tail(20)
top_20.plot(kind="barh", figsize=(16,8))
plt.title('Top 20 sold items')
# %%
print('Number of unique item name: ', len(data['Item Name'].unique()))

# %%
item_freq = data.groupby('Item Name').agg({'Quantity': 'sum'})
item_freq = item_freq.sort_values(by=['Quantity'])
top_20 = item_freq.head(20)
top_20.plot(kind="barh", figsize=(16,8))
plt.title('Least 20 sold items')
# %%
data1 = data ["Total Price"].mean()
print(data1)
# %%
