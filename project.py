# %%[markdown]

#############
## Imports ##
#############

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_log_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, learning_curve
import datetime
from IPython.display import display
import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from mlxtend.frequent_patterns import association_rules, apriori


##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

# %%

##########################
## Set Display Settings ##
##########################

# DF Settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

# Select Color Palette
sns.set_palette('Set2')

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#
# %%
###############
## Load Data ##
###############

data = pd.read_csv("restaurant-1-orders.csv",  parse_dates=['Order Date'])

# %%

# Show number of obs and display full restaurant-1-orders head
print('\nShow head and number of observarions in FULL Restaurant-1-orders data set...\n')
print(f'Restaurant-1-orders observations: {len(data)}')
display(data.head().style.set_sticky(axis="index"))

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

# %%

################
## Clean Data ##
################

# Data Cleaning
# checking Na value
data.isna().sum()

# %%
# Use info() function to print full summary of the dataframe.
# describe() function is used to generate descriptive statistics
# data.shape
data.info()
data.describe()

# %%
# Added new column Total Price
data["Total Price"] = data["Product Price"] * data["Quantity"]
data

# %%
# Frequency for Top 20 sold items

item_freq = data.groupby('Item Name').agg({'Quantity': 'sum'})
item_freq = item_freq.sort_values(by=['Quantity'])
top_20 = item_freq.tail(20)
top_20.plot(kind="barh", figsize=(16, 8))
plt.title('Top 20 sold items')

# %%

top_20.plot(kind="pie", figsize=(16, 8), subplots=True, legend=None)
plt.title('Top 20 sold items')


# Pie diagram for the top 20 items that are being sold based on their quantity.
# %%
print('Number of unique item name: ', len(data['Item Name'].unique()))

# %%
# Frequency For Least 20 Sold Items

item_freq = data.groupby('Item Name').agg({'Quantity': 'sum'})
item_freq = item_freq.sort_values(by=['Quantity'])
top_20 = item_freq.head(20)
top_20.plot(kind="barh", figsize=(16, 8))
plt.title('Least 20 sold items')

# %%
# Getting the average of Total price column with maximum and minimum Total price of orders
data1 = data["Total Price"].mean()
print(data1)
data["Total Price"].max()
data["Total Price"].min()


# %%
# How much people pay for each day in the week
# How much people are paying during thr weeks of the month
print("Daily:\n", data.groupby(
    [pd.Grouper(key='Order Date', freq='D')])['Total Price'].sum().mean())
print("Weekly:\n", data.groupby(
    [pd.Grouper(key='Order Date', freq='W-MON')])['Total Price'].sum().mean())
print("Monthly:\n", data.groupby(
    [pd.Grouper(key='Order Date', freq='M')])['Total Price'].sum().mean())
# %%
# the data has no Null values

data = data.dropna()
data = data.loc[data['Order Date'] >= '2016-08-01']
# %%

# Investigating average order volume by periods
# print("Daily:\n", data.groupby(
#     [pd.Grouper(key='Order Date', freq='D')])['Quantity'].sum().mean())
# print("Weekly:\n", data.groupby(
#     [pd.Grouper(key='Order Date', freq='W-MON')])['Quantity'].sum().mean())
# print("Monthly:\n", data.groupby(
#     [pd.Grouper(key='Order Date', freq='M')])['Quantity'].sum().mean())

# %%
# create relevant Database df1 for total and df2 for bombay aloo

df = data[['Order Date', 'Quantity']]
df2 = data[data['Item Name'] == 'Bombay Aloo']
df2 = df2[['Order Date', 'Quantity']]
df = df.groupby([pd.Grouper(key='Order Date', freq='W-MON')]
                )['Quantity'].sum().reset_index().sort_values('Order Date')
df2 = df2.groupby([pd.Grouper(key='Order Date', freq='W-MON')]
                  )['Quantity'].sum().reset_index().sort_values('Order Date')
# Add Seasonality features
df['Week'] = df['Order Date'].dt.isocalendar().week
df['Month'] = df['Order Date'].dt.month
df2['Week'] = df2['Order Date'].dt.isocalendar().week
df2['Month'] = df2['Order Date'].dt.month
# Add past volume features
for i in range(1, 15):
    label = "Quantity_" + str(i)
    df[label] = df['Quantity'].shift(i)
    df2[label] = df2['Quantity'].shift(i)
    label = "Average_" + str(i)
    df[label] = df['Quantity'].rolling(i).mean()
    df2[label] = df2['Quantity'].rolling(i).mean()
df = df.dropna()
df2 = df2.dropna()
# %%
print(df)
# %%
print(df2)
# %%
# one hot encode df using pandas get_dummies
for column in ['Week', 'Month']:
    tempdf = pd.get_dummies(df[column], prefix=column)
    df = pd.merge(
        left=df,
        right=tempdf,
        left_index=True,
        right_index=True,
    )
    df = df.drop(columns=column)
df.shape
# %%
# one hot encode df2 using pandas get_dummies
for column in ['Week', 'Month']:
    tempdf = pd.get_dummies(df2[column], prefix=column)
    df2 = pd.merge(
        left=df2,
        right=tempdf,
        left_index=True,
        right_index=True,
    )
    df2 = df2.drop(columns=column)
df.shape
# %%
# 143 rows so we split the data up to  row 107 for train and test sets for df
train = df[:107].drop('Order Date', axis=1)
test = df[107:].drop('Order Date', axis=1)
xtrain = train.drop(['Quantity'], axis=1)
xtest = test.drop(['Quantity'], axis=1)
ytrain = train['Quantity']
ytest = test['Quantity']

# %%
# 143 rows so we split the data up to  row 107 for train and test sets for df2
train2 = df2[:107].drop('Order Date', axis=1)
test2 = df2[107:].drop('Order Date', axis=1)
xtrain2 = train2.drop(['Quantity'], axis=1)
xtest2 = test2.drop(['Quantity'], axis=1)
ytrain2 = train2['Quantity']
ytest2 = test2['Quantity']

# %%
#  in the abscence of more data length, and more explanatory variables,
# I have chosen XGBoost (it can handle sparcity without needing to convert to CSR and can handle multiple regressors).
# We could run this model and save the results through a loop, to run it across multipel epochs for further into the
#  future predictions. But as per our problem statement, we are going to focus on just the next week at a time
# (in a real life situation, as the busniess owner, we may run this at the end of each week, to get orders and staff prepared over the weekend for next week).

# Model for df

model = XGBRegressor(n_estimators=500, learning_rate=0.01)
eval_set = [(xtrain, ytrain)]
model.fit(xtrain, ytrain, eval_metric="rmsle", eval_set=eval_set,
          early_stopping_rounds=20, verbose=False)
ypred = model.predict(xtest)

# %%
# Model for df2
model2 = XGBRegressor(n_estimators=500, learning_rate=0.01)
eval_set = [(xtrain2, ytrain2)]
model2.fit(xtrain2, ytrain2, eval_metric="rmsle", eval_set=eval_set,
           early_stopping_rounds=20, verbose=False)
ypred2 = model2.predict(xtest2)

# %%
# Finally, I evaluate both the models: both on graphs, and then using the relevant metrics. We optmised scoring
# for root mean square log error as it standardises for when a quantity is already high and thus the absolute percentage change is low

# First we add the results to our original dataframe, after first aligning the indexes

# df
ypred = pd.Series(ypred)
eval_df = df[107:].reset_index(drop=True)
eval_df['ypred'] = ypred
eval_df = eval_df[['Order Date', 'Quantity', 'ypred']]
eval_df.head()

# df2
ypred2 = pd.Series(ypred2)
eval_df2 = df2[107:].reset_index(drop=True)
eval_df2['ypred'] = round(ypred2)
eval_df2 = eval_df2[['Order Date', 'Quantity', 'ypred']]
eval_df2.head()

# %%
# And Now we plot the results of the train vs test sets
# df
plt.figure(figsize=(20, 8))
plt.plot(eval_df['Order Date'], eval_df['Quantity'], label="Actual Quanitity")
plt.plot(eval_df['Order Date'], eval_df['ypred'],
         color='red', label='Predicted Quantity')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.legend()
plt.title('Total Sales')

# df2
plt.figure(figsize=(20, 8))
plt.plot(eval_df2['Order Date'], eval_df2['Quantity'],
         label="Actual Quanitity")
plt.plot(eval_df2['Order Date'], eval_df2['ypred'],
         color='red', label='Predicted Quantity')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.legend()
plt.title('Bombay Aloo Sales')

# %%
# Lastly metrics mean_absolute_error, r2_score, mean_squared_log_error
# df
print("Metrics for Total Sale\n")
print("Mean Absolute Error:\n", mean_absolute_error(ytest, ypred))
print("R Squared:\n", r2_score(ytest, ypred))
print("Mean Squared Log Error:\n", mean_squared_log_error(ytest, ypred))

# df2
print("\n")
print("Metrics for Bombay Aloo Sales\n")
print("Mean Absolute Error:\n", mean_absolute_error(ytest2, ypred2))
print("R Squared:\n", r2_score(ytest2, ypred2))
print("Mean Squared Log Error:\n", mean_squared_log_error(ytest2, ypred2))

# The clearest metric is the RMSE. This shows clearly that the model for Total sales worked really quite well.
#  The Bombay Aloo model however was less successful, most likely due to the far smaller quantity per week.
# So the Total sales model looks like it would be good to start deploying, but perhaps the Bombay Aloo model
# should be rethought/scrapped, or alternatively changed to a monthly model, which may improve accuracy (though may not be as useful to the business)
# As always, improvement could be had with cross validation, more data, hyperparameter optimisation, and
# possibly tryign some of the other models mentioned above)
# %%


# Orders by time of day
# We are going to calculate the average number of orders that are placed in each hour
# of the day to give us an idea of ​​when the demand is greatest.
# Add column with the time of the order
data['hour'] = data['Order Date'].dt.hour
data.sample(5)
# %%
data['date'] = data['Order Date'].dt.strftime(
    '%y/%m/%d')  # Add column with date
data.sample(5)

# The way we will calculate the average orders per hour is as follows:
# For a specified hour, we will calculate the number of orders that were taken at that hour considering
# the average per day.
# %%


def avg_hour(hour):
    by_hour = data[data['hour'] == hour]
    avg = len(by_hour['Order Number'].unique()) / len(data['date'].unique())
    return avg


hours = pd.DataFrame(sorted(data['hour'].unique()))

hours.rename(columns={0: 'hour'}, inplace=True)
hours['Average orders'] = hours['hour'].apply(avg_hour)

hours.set_index('hour', inplace=True)
hours.head()
# %%
hours.plot.bar(figsize=(11, 6), rot=0)
plt.xlabel('Hour')
plt.title('Average number of orders by hour of the day')
# As can be seen, the hours at which the greatest number of orders are made on average
# are 5, 6, and 7:00 p.m., with a peak at 6:00 p.m.
# %%
# Orders by day of the week
# We are going to do the same analysis as before but this time considering the
# different days of the week.
# Column with the name of the day
data['day'] = data['Order Date'].dt.day_name()
data.sample(5)
# %%


def by_day(day):
    data_day = data[data['day'] == day]
    avg = len(data_day['Order Number'].unique()) / \
        len(data_day['date'].unique())
    return(avg)


days = pd.DataFrame(['Monday', 'Tuesday', 'Wednesday',
                    'Thursday', 'Friday', 'Saturday', 'Sunday'])
days.rename(columns={0: 'day'}, inplace=True)
days['avg_orders'] = days['day'].apply(by_day)
days

# %%
plt.bar(days['day'], days['avg_orders'])
plt.xlabel('Day of week')
plt.ylabel('Average number of orders')
plt.title('Average orders by day of week')
plt.xticks(rotation=90)
# The graph shows that the day with the highest average number of orders is Saturday.
# Interestingly, more averages are performed on Friday than on Sundays.

# %%
# We will visualize sales over time considering monthly time periods
print('First Sale: ', data['Order Date'].min())
print('Last Sale: ', data['Order Date'].max())
# %%
# We will then consider the dates between January 2016 and December 2019.

months = []

for year in range(2016, 2020):
    for month in range(1, 13):
        d = datetime.date(year, month, 1)
        months.append(d)

monthly = pd.DataFrame(months)
monthly.rename(columns={0: 'month'}, inplace=True)
monthly.head()

# Now we will assign each month its total sales.

# %%


def sales_month(date):
    year_month = date.strftime('%y/%m')
    data1 = data[data['date'].str[:5] == year_month].copy()
    total = (data1['Quantity'] * data1['Product Price']).sum()
    return(total)


monthly['total'] = monthly['month'].apply(sales_month)
monthly.head()
# %%
#################### can we have the dates 90 degree ###
plt.plot(monthly['month'], monthly['total'])
plt.xlabel('Date')
plt.ylabel('Total sales (USD)')
plt.title('Total monthly sales')
# You can see that monthly sales had been growing up to a point in the middle of
# 2019, where they suffered a big drop. Let's identify this point.
# %%
monthly[monthly['month'] >= datetime.date(2019, 1, 1)]
# The month in which sales fell was August 2019.
# %%

# We are going to visualize the distribution of the cost of the orders to the restaurant.
order_total = data[['Order Number', 'Quantity', 'Product Price']].copy()

############ I belive we to answer the question of average restaurant prices we should not do math? ###
order_total['total'] = order_total['Quantity'] * order_total['Product Price']


# very intereting!! Can we invistigate the very high prices?!
# Add the order price
order_totals = order_total.groupby('Order Number').agg({'total': 'sum'})
plt.boxplot(order_totals['total'])
plt.title('Order price distribution')


# %%
p_95 = order_totals['total'].describe(percentiles=[0.95])['95%']
print('95% of the orders are less than or equal to {percentile} USD'.format(
    percentile=p_95))
# 95% of the orders are less than or equal to 62.2 USD
# Let's consider the distribution for the total price of orders less than 63 USD.

# %%
plt.boxplot(order_totals[order_totals['total'] < 63]['total'])
plt.title('Order total USD')
plt.ylabel('USD')

# %%
sns.distplot(order_totals[order_totals['total'] < 63], bins=20)
plt.title('Order price distribution')

# %%

# restaurant prices and restaurant quantity of each item
item_freq2 = data.groupby('Item Name').agg(
    {'Quantity': 'sum', 'Product Price': 'mean'})
item_freq2.mean()
item_freq2.max()
item_freq2.min()

# plot
sns.scatterplot(item_freq2, x='Product Price', y='Quantity')
plt.title('Item price VS item quantity')
plt.show()

# delete outliers of quantity and price
items = item_freq2[(item_freq2['Product Price'] <= 14.)
                   & (item_freq2['Quantity'] <= 7000.)]

sns.scatterplot(items, x='Product Price', y='Quantity')
plt.title('Item price VS item quantity')
plt.show()
# %%
# is theer carrolation between the product price and number of total orders of each item?
#  Pearson correlation coefficient
corr, _ = pearsonr(items['Product Price'], items['Quantity'])
print('Pearsons correlation: %.3f' % corr)
print('''The two varibles have low carrolation of .4 that means people decition of bying an item is not heavely based on the price.
      However, the test shows a negative result therefore we could say that the generally when the price of an item goes up, probably
      the number of orders will go down''')

# Another test for no-liner relationship or not Normal Distribution varibles
# Spearman correlation coefficient
corr, _ = spearmanr(items['Product Price'], items['Quantity'])
print('Spearmans correlation: %.3f' % corr)
print('There is .5 carrolation between the two samples however, it is not strong and we stick with our previous clime')

# %%
#####  preparing data  #######
# Take only the needed columns
res2 = data[['Order Number', 'Item Name', 'Quantity']].copy()

# Create cateogries for each item that has different flavors
res2['Item Name'] = res2['Item Name'].apply(lambda x: 'Naan' if 'Naan' in x else 'Sauce' if 'Sauce' in x else 'Papadum' if 'Papadum' in x else 'Salad' if 'Salad' in x else 'Balti' if 'Balti' in x
                                            else 'Rice' if 'Rice' in x else 'Balti' if 'Balti' in x else 'Bhajee' if 'Bhajee' in x else 'Bhajee' if 'Bhaji' in x else 'Mushroom' if 'Mushroom' in x else
                                            'Chutney' if 'Chutney' in x else'Pasanda' if 'Pasanda' in x else 'Biryani' if 'Biryani' in x else 'Korma' if 'Korma' in x else 'Aloo' if 'Aloo' in x else
                                            'Curry' if 'Curry' in x else 'Sheek' if 'Sheek' in x else 'Samosa' if 'Samosa' in x else 'Hari Mirch' if 'Hari Mirch' in x else 'Madras' if 'Madras' in x
                                            else 'wine' if 'wine' in x else 'Lemonade' if 'Lemonade' in x else 'Water' if 'Water' in x else 'COBRA' if 'COBRA' in x else 'Coke' if 'Coke' in x else
                                            'Karahi' if 'Karahi' in x else 'Jalfrezi' if 'Jalfrezi' in x else 'Bhuna' if 'Bhuna' in x else 'Dupiaza' if 'Dupiaza' in x else 'Methi' if 'Methi' in x
                                            else 'Lal Mirch' if 'Lal Mirch' in x else 'Shashlick' if 'Shashlick' in x else 'Shashlick' if 'Shaslick' in x else 'Sizzler' if 'Sizzler' in x else
                                            'Dall' if 'Dall' in x else 'Sylhet' if 'Sylhet' in x else 'Mysore' if 'Mysore' in x else 'Puree' if 'Puree' in x else 'Paratha' if 'Paratha' in x else
                                            'Chaat' if 'Chaat' in x else 'Achar' if 'Achar' in x else 'Vindaloo' if 'Vindaloo' in x else 'Dhansak' if 'Dhansak' in x else 'Haryali' if 'Haryali' in x
                                            else 'Rogon' if 'Rogon' in x else 'Hazary' if 'Hazary' in x else 'Roshni' if 'Roshni' in x else 'Jeera' if 'Jeera' in x else 'Rezala' if 'Rezala' in x
                                            else 'Bengal' if 'Bengal' in x else 'Bengal' if 'Butterfly' in x else 'Pathia' if 'Pathia' in x else 'Masala' if 'Masala' in x else 'Paneer' if 'Paneer' in x
                                            else 'Saag' if 'Saag' in x else 'Tandoori' if 'Tandoori' in x else 'Tikka' if 'Tikka' in x else 'Chilli Garlic' if 'Chilli' in x else 'Chapati' if 'Chapati' in x
                                            else 'Pickle' if 'Pickle' in x else 'Pakora' if 'Pakora' in x else 'Fries' if 'Fries' in x else 'Starters' if 'Starter' in x else 'Raitha' if 'Raitha' in x
                                            else 'Rolls' if 'Roll' in x else 'Butter Chicken' if 'Butter' in x else 'Persian')

# check if it worked
col = res2['Item Name'] == 'wine'
res2[col]
res2[res2['Order Number'] == 11217]

# Group order number so each order has its ouwn set of items
item_list = (res2.groupby(['Order Number', 'Item Name'])['Quantity']
             .min().unstack().reset_index().fillna(0)
             .set_index('Order Number'))

# create one hot incoding


def hot_encode(x):
    if(x <= 0):
        return 0
    if(x >= 1):
        return 1


# Encoding the datasets
sparse_df_items = item_list.applymap(hot_encode)

# Check if the order number has the same values as before applying the function
sparse_df_items.loc[11217][["Naan", "Bhajee",
                            "Rice", "Curry", "Masala", "Balti"]]

# ML section
# %% [markdown]
#### Undersatnding Apriori parameters ######
# Apriori is a popular algorithm [1] for
# extracting frequent itemsets with applications
# in association rule learning. The apriori algorithm has been designed
# to operate on databases containing transactions, such as purchases by customers
# of a store. An itemset is considered as "frequent" if it meets a user-specified support
# threshold. For instance, if the support threshold is set to 0.5 (50%),
# a frequent itemset is defined as a set of items that occur together
# in at least 50% of all transactions in the database.
# from http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/

# doing the calculation
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
# %%
# Do ML to claculate probability of association
frequent_itemsets = apriori(
    sparse_df_items, min_support=0.02209, use_colnames=True, verbose=1)

# These are the companations of orders we have on the chance of %2 and more
frequent_itemsets.shape
frequent_itemsets.head()
association = association_rules(
    frequent_itemsets, metric="lift", min_threshold=1)

# I have to check the left rule where it says
# filter the best recommendations we will use the highest confidence value for each antecedent. We ran with the top 20 most frequent items and got some recommendations
recommendations = association.sort_values(
    ['confidence', 'lift'], ascending=False)
# Explain why we are interrested in lift more than others

# Save it to csv for any future useage instead of running the code file
# recommendations.to_csv('companations.csv')

# We are interested in giving unusual recommenditions to the restaurant
best_item_recommendations2 = recommendations.drop_duplicates(
    subset=['consequents']).head(20)

# Searching for the left, support and confedint of specific companations
# the best companations based on the highest numbers
recommendations[(recommendations['lift'] >= 0.4) &
                (recommendations['confidence'] >= 0.9) &
                (recommendations['support'] >= 0.04)]

# check the probability of a specific item to be with another
recommendations.loc[(recommendations['antecedents'] == {'Korma'}) & (
    recommendations['consequents'] == {'Naan'})]
# check What could come with a specidic item
# We should search for the lowest ordered items
recommendations.loc[recommendations['antecedents'] == {'Korma'}]
# %%
# %%

Min_quantity = data['Quantity'].min()
Max_Quantity = data['Quantity'].max()
print(Min_quantity)
print(Max_Quantity)

# %%
for col in data:
    if col == 'Item Name':
        unique = (data[col].unique())
        print(len(unique))
print(unique)

# %%
Item_unique = pd.DataFrame(unique)
Item_unique.columns = ['Name']
Item_unique.head()

# %%
Item_sorted = Item_unique.sort_values('Name', ignore_index=True)
Item_sorted.head()


# %%
# See if there is any relationship between the product quantity and the total price
sns.regplot(data=data, x="Quantity", y="Total Price")
plt.show()

# You can see that the product price has a linear relationship with the total price,
# and you can make simple predictions
# %%
# Linear Regression Model Prediction

X = data["Quantity"].tolist()
X = np.array(X).reshape(-1, 1)
y = data["Total Price"]
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=0)
model = LinearRegression()
model = model.fit(X_train, y_train)
model.score(X_test, y_test)

# It can be seen that the accuracy is 0.03.
# %%
# draw the learning curve
train_sizes, train_loss, test_loss = learning_curve(
    LinearRegression(), X, y, train_sizes=[0.1, 0.25, 0.5, 0.75, 1])
train_mean = np.mean(train_loss, axis=1)
test_mean = np.mean(test_loss, axis=1)
plt.plot(train_sizes, train_mean, label="Training")
plt.plot(train_sizes, test_mean, label="Cross-validation")
plt.xlabel("Training sizes")
plt.ylabel("score")
plt.legend()
plt.show()

# %%
data["date"] = pd.to_datetime(data['Order Date']).dt.date
data

# %%
# get the values that are unique
unique_data = data.drop_duplicates(subset=["Item Name"])
unique_data

# %%

# Getting the count of the Items that are ordered as a total

count = data.groupby(['Item Name']).count().reset_index()
count = count.iloc[:, :2]
count.columns = ['Item Name', "Count"]
count


# %%

# joining the product price column to the above created data frame

count_1 = count.merge(unique_data, on="Item Name", how="left")
count_1 = count_1[["Item Name", "Count", "Product Price"]]
count_1 = count_1.sort_values(by="Count", ascending=False)
count_1

# %%
sns.scatterplot(x="Count", y="Product Price", data=count_1)
plt.xlim(0, 1000)

# %%
corr, _ = pearsonr(count_1["Product Price"], count_1["Count"])
print('Pearsons correlation: %.3f' % corr)

# Greater the product price lower are the number of orders placed


# %%

date_count = data['date'].nunique()
date_count

# %%

sns.scatterplot(data=date_count, x="date", y="count")

# %%
count_1['average_orders_per_day'] = count_1['Count']/date_count
count_1

# This gives the average number of orders that an item is being ordered in a day.
# %%


# Pairs of best-selling items
# We are going to visualize which are the items that are bought the most together

all_items = list(data['Item Name'].unique())

# association table

associations = pd.DataFrame(index=all_items, columns=all_items)
associations.fillna(0, inplace=True)
associations.iloc[:4, :4]
# %%
# We are going to populate the previous table by counting in each entry the number
# of times that a pair of items was requested in different orders.
orders = data.groupby('Order Number')['Item Name'].apply(
    lambda x: ','.join(x)).reset_index()
orders.rename(columns={'Item Name': 'Order'}, inplace=True)
orders['Order'] = orders['Order'].str.split(',')
orders.head(20)

# %%

# Popular the table
for Order in orders['Order']:
    associations.loc[Order, Order] += 1
# %%
associations.iloc[:4, :4]
# %%
# As this table is very large, we are going to restrict ourselves only to the
# pairs within the top 20 best-selling items.

associations_top = associations.loc[list(top_20.index), list(top_20.index)]

for i in range(associations_top.shape[0]):
    for j in range(i, associations_top.shape[0]):
        associations_top.iloc[i, j] = 0

associations_top.iloc[:5, :5]
# %%
# We will generate a heat map to visually identify which are the most common pairs.
plt.figure(figsize=(12, 8))
plt.title('Common sold together items')
sns.heatmap(associations_top, cmap="Greens", annot=False)
# %%
# data.columns.is_unique
# data.columns.duplicated()
# data.loc[:, ~data.columns.duplicated()]
# # %%
# df.columns.is_unique
# df.columns.duplicated()
# df.loc[:, ~df.columns.duplicated()]
# %%
