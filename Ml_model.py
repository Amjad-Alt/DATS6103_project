# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%%[markdown]'

#%%
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()

res = pd.read_csv('restaurant-1-orders.csv')
res.head
#%%
#####  preparing data  #######
# Take only the needed columns
res2= res[['Order Number', 'Item Name', 'Quantity']].copy()

# Create cateogries for each item that has different flavors  
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
col2 = res2['Item Name']== 'Naan'
res2[col2]
res2[res2['Order Number'] == 11217]

# Group order number so each order has its ouwn set of items
item_list2 = res2.groupby('Order Number')['Item Name'].unique()
# another way of grouping
item_list2 = (res2.groupby(['Order Number', 'Item Name'])['Quantity']
             .min().unstack().reset_index().fillna(0)
             .set_index('Order Number'))
def hot_encode(x):
    if(x<= 0):
        return 0
    if(x>= 1):
        return 1
  
# Encoding the datasets
sparse_df_items2 = item_list2.applymap(hot_encode)

# sperade items names each in one column
# Set TRUE if the item is order in that order item 
oht_orders2 = te.fit(item_list2).transform(item_list2, sparse=True)

# convert it to dataframe using columns from the previous code
sparse_df_items2 = pd.DataFrame.sparse.from_spmatrix(oht_orders2, columns=te.columns_)
sparse_df_items2['Naan']
# replace True by 1
sparse_df_items2 = sparse_df_items2.astype('int')

############### ML section 
# %% [markdown]
#### Undersatnding Apriori parameters ###### 
#Apriori is a popular algorithm [1] for 
#extracting frequent itemsets with applications 
#in association rule learning. The apriori algorithm has been designed 
#to operate on databases containing transactions, such as purchases by customers 
#of a store. An itemset is considered as "frequent" if it meets a user-specified support
#threshold. For instance, if the support threshold is set to 0.5 (50%), 
#a frequent itemset is defined as a set of items that occur together 
#in at least 50% of all transactions in the database.
# from http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/

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

# from https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/
# from https://www.kdnuggets.com/2016/04/association-rules-apriori-algorithm-tutorial.html
 #%%
# Do ML to claculate probability of association 
frequent_itemsets2 = apriori(sparse_df_items2, min_support=0.02209, use_colnames=True, verbose=1)

#These are the companations of orders we have on the chance of %2 and more
frequent_itemsets2.shape
frequent_itemsets2.head()
market_basket_rules = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)

# I have to check the left rule where it says 
#filter the best recommendations we will use the highest confidence value for each antecedent. We ran with the top 20 most frequent items and got some recommendations
best_item_recommendations = market_basket_rules.sort_values(['confidence', 'lift'],ascending=False)
#Explain why we are interrested in lift more than others

# Save it to csv for any future useage instead of running the code file
#best_item_recommendations.to_csv('companations.csv')

#We are interested in giving unusual recommenditions to the restaurant 
best_item_recommendations3 = best_item_recommendations.drop_duplicates(subset=['consequents']).head(20)

# Searching for the left, support and confedint of specific companations
# the best companations based on the highest numbers 
best_item_recommendations[ (best_item_recommendations['lift'] >= 0.4) &
                           (best_item_recommendations['confidence'] >= 0.9) &
                           (best_item_recommendations['support'] >= 0.04)]

# check the probability of a specific item to be with another       
best_item_recommendations.loc[(best_item_recommendations['antecedents'] == 'Curry') & (best_item_recommendations['consequents'] == 'Naan')]
# %%
