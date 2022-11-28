# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%%[markdown]'

#%%
import numpy as np
import pandas as pd

###### prepare the data for machine learning
res = pd.read_csv('restaurant-1-orders.csv')
res.head
#take only the order number and the item name
res1= res[['Order Number', 'Item Name']].copy()
#make category
#res1['item_cat'] = res1['Item Name'].astype('category')
#res2 = res1.groupby('Order Number')['item_cat'].unique()
#res3 = pd.DataFrame(res2)
#code numbers
res1['Item Name_num'] = res1['item_cat'].cat.codes
res1.head
#spread the column into multiple 0,1 columns
df = pd.get_dummies(res1.set_index('Order Number'), prefix='', prefix_sep='').max(level=0).reset_index()
#take only the codded column of orders and the order number
res2= res1[['Order Number', 'Item Name_num']].copy()
df2 = df.groupby(["Order Number"])
#drop_item num
df3 = df.drop('Item Name_num', axis=1)
#I have to make sure that all values are 0 and 1
df3.to_csv('out3.csv')  

######################################################
#pip install mlxtend
from mlxtend.frequent_patterns import apriori

df4 = pd.read_csv('out3.csv',index_col=0)
df5 = df4.drop('Order Number', axis=1)
df6 = df5.iloc[:4000, :]
rules = apriori(df6, min_support = 0.003, use_colnames=True)
print(list(rules))

#rules1 = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
#rules1.head()
#RelationRecord(items=frozenset({'avocado', 'spaghetti', 'milk'}), support=0.003332888948140248, ordered_statistics=[OrderedStatistic(items_base=frozenset({'avocado', 'spaghetti'}), items_add=frozenset({'milk'}), confidence=0.41666666666666663, lift=3.215449245541838)]),
#################################################################