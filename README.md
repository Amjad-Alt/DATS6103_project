
# Project Proposal
We are analyzing and applying machine learning algorithms to an *Indian takeaway restaurant’s dataset* placed in London, UK in 2019. 
 
##     Research topic
Raising the profit of an *Indian takeaway restaurant* by giving marketing suggestions of association supported by data.
 
##     SMART question(s)
- How likely product A will be ordered with product B? 
- What is the busiest time in the day/week? 
- What is the most ordered product and the least ordered? 
- What is the average payment for orders? 
- What is the average quantity of every product per day? 
- Is there a correlation between product price and the number of total orders of each product?

##     Source of data set and observations
The dataset has more than `7000` observations and `6` columns. 
1.**Order Number:** receipt number.
2.**Order Date:** the day and time of the order.
3.**Item Name:** the name of the order in the menu.
4.**Quantity:** the number of orders per item for the receipt.
5.**Product Price:** item price without counting the quantity.
6.**Total products:** total number of items per receipt.

Data source: [here](https://www.kaggle.com/datasets/henslersoftware/19560-indian-takeaway-orders?select=restaurant-1-orders.csv)

##    Modeling method
To determine the degree of association between two objects, we will use `Apriori` algorithm to extract frequent item sets.
