# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%%[markdown]'

#%%
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()

###### Very over detailed prbability of association between items ##########
res = pd.read_csv('restaurant-1-orders.csv')
res.head
#take only the order number and the item name
res1= res[['Order Number', 'Item Name']].copy()
#group items each to its order number
item_list = res1.groupby('Order Number')['Item Name'].unique()
# sperade items each in one column and transform the values to TRUE if that item is there or 0 if it not there
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

frequent_itemsets2 = apriori(sparse_df_items2, min_support=0.02209, use_colnames=True, verbose=2)
#These are the companations of orders we have on the chance of %2 and more
frequent_itemsets2.shape
frequent_itemsets2.head()
market_basket_rules = association_rules(frequent_itemsets2, metric="lift", min_threshold=2)
# add a column length to see how many items compined 
frequent_itemsets2['length'] = frequent_itemsets2['itemsets'].apply(lambda x: len(x))
frequent_itemsets2.groupby('length').describe()
print('After grouping items there are more compination of two items 116 and three items 112')

# I have to check the left rule where it says 
#filter the best recommendations we will use the highest confidence value for each antecedent. We ran with the top 20 most frequent items and got some recommendations
best_item_recommendations = market_basket_rules.sort_values(['confidence','lift'],ascending=False).drop_duplicates(subset=['consequents'])
top_20_frequence_items = frequent_itemsets2.sort_values('support',ascending=False).head(20)['itemsets']
best_item_recommendations[best_item_recommendations['consequents'].isin(top_20_frequence_items)]

# %% [markdown]
#### Undersatnding Apriori parameters ###### 
# The first parameter is the list of list that you want to extract rules from.
# The second parameter is the 'min_support' parameter that is used to select the items with support values greater than the value specified by the parameter.
# the 'min_confidence' parameter filters those rules that have confidence greater than the confidence threshold specified by the parameter. 
# the 'min_lift' parameter specifies the minimum lift value for the short listed rules.
# Finally, the 'min_length' parameter specifies the minimum number of items that you want in your rules.

### doing the calculation 
# Let's suppose that we want rules for only those items that are purchased at least 5 times a day, or 7 x 5 = 35 times in one week, since our dataset is for a one-week time period. 
# The support for those items can be calculated as 35/7500 = 0.0045.
# The minimum confidence for the rules is 20% or 0.2.
# we specify the value for lift as 3 and finally min_length is 2 since we want at least two products in our rules.
#### Understanding Apriori result #####
# Measure 1: Support. This says how popular an itemset is, as measured by the proportion of transactions in which an itemset appears.
# Support refers to the default popularity of an item and can be calculated by finding number of transactions containing a particular item divided by total number of transactions.
# For instance if out of 1000 transactions, 100 transactions contain Ketchup then the support for item Ketchup can be calculated as: Support(Ketchup) = 100/1000 = 10%
# Support(Ketchup) = (Transactions containingKetchup)/(Total Transactions)

# Measure 2: Confidence. This says how likely item B is purchased when item A is purchased
# Confidence refers to the likelihood that an item B is also bought if item A is bought.
# This is because it only accounts for how popular apples are, but not beers. If beers are also very popular in general, there will be a higher chance that a transaction containing apples will also contain beers 
#  It can be calculated by finding the number of transactions where A and B are bought together, divided by total number of transactions where A is bought. 
# Confidence(A→B) = (Transactions containing both (A and B))/(Transactions containing A)
     
      
# Measure 3: Lift. This says how likely item B is purchased when item A is purchased, while controlling for how popular item B is.
# refers to the increase in the ratio of sale of B when A is sold.
# Lift(Burger→Ketchup) = (Confidence (Burger→Ketchup))/(Support (Ketchup))
# Lift basically tells us that the likelihood of buying a Burger and Ketchup together is 3.33 times more than the likelihood of just buying the ketchup. A Lift of 1 means there is no association between products A and B. 
# Finally, Lift of less than 1 refers to the case where two products are unlikely to be bought together.
                
# %%
