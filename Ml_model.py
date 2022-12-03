# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%%[markdown]'

#%%
import numpy as np
import pandas as pd

###### Very over detailed prbability of association between items ##########
res = pd.read_csv('restaurant-1-orders.csv')
res.head
#take only the order number and the item name
res1= res[['Order Number', 'Item Name']].copy()
#group items each to its order number
item_list = res1.groupby('Order Number')['Item Name'].unique()
# sperade items each in one column and transform the values to TRUE if that item is there or 0 if it not there
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
oht_orders = te.fit(item_list).transform(item_list, sparse=True)
# convert it to dataframe using columns from the TransactionEncoder
sparse_df_items = pd.DataFrame.sparse.from_spmatrix(oht_orders, columns=te.columns_)
# replace True by 1
sparse_df_items = sparse_df_items.astype('int')
# make a sperate csv.file to use 
#sparse_df_items.to_csv('data_2.csv')
#As a threshold for the minimum frequency of a set of items(the support metric), we used the percentage of the average/unique order frequency, which is 2.22% and max len of set of items equals 10.
from mlxtend.frequent_patterns import apriori, association_rules
frequent_itemsets = apriori(sparse_df_items, min_support=0.02209, use_colnames=True, verbose=1)
#These are the companations of orders we have on the chance of %2 and more
frequent_itemsets.shape
frequent_itemsets.head()
# add a column length to see how many items compined 
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets.groupby('length').describe()
print('there are 88 companation of two items, 28 combanations of three items and only one of four items')
print('according to Support average there are more frequency of one item and it become less with more items')
#select results to our desired goal like in here we want two items that has the frequency of 3%
frequent_itemsets[ (frequent_itemsets['length'] == 2) &
                   (frequent_itemsets['support'] >= 0.03) ]
# check the probability of a specific item to be with another 
frequent_itemsets[ frequent_itemsets['itemsets'] == {'Plain Papadum', 'Mint Sauce'} ]



############## a little general probability by groubing items ############
res2= res[['Order Number', 'Item Name']].copy()

# Grouping items if they are the same except different flavors  
res2['Item Name'] = res2['Item Name'].apply( lambda x: 'Naan' if 'Naan' in x else 'Sauce' if 'Sauce' in x else 'Papadum' if 'Papadum' in x else 'Salad' if 'Salad' in x else 'Balti' if 'Balti' in x 
                                            else 'Rice' if 'Rice' in x else 'Balti' if 'Balti' in x else 'Bhajee' if 'Bhajee' in x else 'Bhajee' if 'Bhaji' in x else 'Mushroom' if 'Mushroom' in x else
                                            'Chutney' if 'Chutney' in x else'Pasanda' if 'Pasanda' in x else 'Biryani' if 'Biryani' in x else 'Korma' if 'Korma' in x else 'Aloo' if 'Aloo' in x else 
                                            'Curry' if 'Curry' in x else 'Sheek' if 'Sheek' in x else 'Samosa' if 'Samosa' in x else 'Hari Mirch' if 'Hari Mirch' in x else 'Madras' if 'Madras' in x 
                                            else 'wine' if 'wine' in x else 'Lemonade' if 'Lemonade' in x else 'Water' if 'Water' in x else 'COBRA' if 'COBRA' in x else 'Coke' if 'Coke' in x else 
                                            'Karahi' if 'Karahi' in x else 'Jalfrezi' if 'Jalfrezi' in x else 'Bhuna' if 'Bhuna' in x else 'Dupiaza' if 'Dupiaza' in x else 'Methi' if 'Methi' in x 
                                            else 'Lal Mirch' if 'Lal Mirch' in x else 'Shashlick' if 'Shashlick' in x else 'Shashlick' if 'Shaslick' in x else 'Sizzler' if 'Sizzler' in x else 
                                            'Dall' if 'Dall' in x else 'Sylhet' if 'Sylhet' in x  else 'Mysore' if 'Mysore' in x else 'Puree' if 'Puree' in x else 'Paratha' if 'Paratha' in x else 
                                            'Chaat' if 'Chaat' in x else 'Achar' if 'Achar' in x else 'Vindaloo' if 'Vindaloo' in x else  'Dhansak' if 'Dhansak' in x else 'Haryali' if 'Haryali' in x 
                                            else 'Rogon' if 'Rogon' in x  else 'Hazary' if 'Hazary' in x else 'Roshni' if 'Roshni' in x else 'Jeera' if 'Jeera' in x else 'Rezala' if 'Rezala' in x 
                                            else 'Bengal' if 'Bengal' in x  else 'Bengal' if 'Butterfly' in x  else 'Pathia' if 'Pathia' in x else 'Masala' if 'Masala' in x else 'Paneer' if 'Paneer' in x
                                            else 'Saag' if 'Saag' in x else 'Tandoori' if 'Tandoori' in x  else 'Tikka' if 'Tikka' in x else 'Chilli Garlic' if 'Chilli' in x else 'Chapati' if 'Chapati' in x
                                            else 'Pickle' if 'Pickle' in x else 'Pakora' if 'Pakora' in x  else 'Fries' if 'Fries' in x else 'Starters' if 'Starter' in x else 'Raitha' if 'Raitha' in x
                                            else 'Rolls' if 'Roll' in x else 'Butter Chicken' if 'Butter' in x  else 'Persian') 

# check if it worked
col = res2['Item Name']== 'wine'
res2[col]
col2 = res2['Item Name']== 'Rolls'
res2[col2]
#group items each to its order number
item_list2 = res2.groupby('Order Number')['Item Name'].unique()
# sperade items each in one column and transform the values to TRUE if that item is there or 0 if it not there
oht_orders2 = te.fit(item_list2).transform(item_list2, sparse=True)
# convert it to dataframe using columns from the TransactionEncoder
sparse_df_items2 = pd.DataFrame.sparse.from_spmatrix(oht_orders2, columns=te.columns_)
# replace True by 1
sparse_df_items2 = sparse_df_items2.astype('int')
frequent_itemsets2 = apriori(sparse_df_items2, min_support=0.02209, use_colnames=True, verbose=1)
#These are the companations of orders we have on the chance of %2 and more
frequent_itemsets2.shape
frequent_itemsets2.head()
# add a column length to see how many items compined 
frequent_itemsets2['length'] = frequent_itemsets2['itemsets'].apply(lambda x: len(x))
frequent_itemsets2.groupby('length').describe()
print('After grouping items there are more compination of two items 116 and three items 112')

# get only rules that have a probability of buying the antecedents and consequents in the same order.
market_basket_rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)
market_basket_rules2.head()
market_basket_rules2.groupby('antecedents').size().sort_values(ascending=False)

# I have to check the left rule where it says 
#filter the best recommendations we will use the highest confidence value for each antecedent. We ran with the top 20 most frequent items and got some recommendations
best_item_recommendations = market_basket_rules2.sort_values(['confidence','lift'],ascending='False').drop_duplicates(subset=['antecedents'])
top_20_frequence_items = frequent_itemsets2.sort_values('support',ascending=False).head(20)['itemsets']
best_item_recommendations[best_item_recommendations['antecedents'].isin(top_20_frequence_items)]

#Check this method in the Apeori 
#RelationRecord(items=frozenset({'avocado', 'spaghetti', 'milk'}), support=0.003332888948140248, ordered_statistics=[OrderedStatistic(items_base=frozenset({'avocado', 'spaghetti'}), items_add=frozenset({'milk'}), confidence=0.41666666666666663, lift=3.215449245541838)]),
#################################################################
# %%
