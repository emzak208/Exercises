# Assessment

#Instructions
#To Run the script, enter the following command in Terminal
# $ python are_we_going_to_survive.py original_purchase_order.csv next_purchase_order.csv customer_features.csv product_features.csv last_month_assortment.csv next_month_assortment.csv


#Import libraries
import pandas as pd 
import numpy as np
from functools import reduce
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")


# In[20]:


#Some Functions to.. 
#Calculate Shipping Costs
def shipping_cost(x):
    return 0.6 if x == 1 else 1.2

#Calculate Sales
def sales(x, y):
    return y if x == 1 else 0

#Calculate Total Cost per product
def total_costs(df):
    df['cost_by_product'] = df['quantity_purchased']*df['cost_to_buy']
    
def ROI(a,b,c,d,e,f):
    print ("Yes") if ((e+f)-(a+b+c+d))>=0 else print("No")


# In[21]:


#cd '/Users/axjx/Downloads/assessment'


# In[22]:


#Read in the data
last_month_assortment = pd.read_csv('last_month_assortment.csv')
last_purchase_order = pd.read_csv('original_purchase_order.csv')
next_month_assortment = pd.read_csv('next_month_assortment.csv')
next_purchase_order = pd.read_csv('next_purchase_order.csv')
customer_features = pd.read_csv('customer_features.csv')
prod_features = pd.read_csv('product_features.csv')


# Data Preprocessing

# Customer Features

# In[23]:


#How many levels are in each categorical variables in Customer Features?
# for cat in ['age_bucket', 'is_returning_customer', 'favorite_genres']: 
#     print("Number of levels in category '{0}': \b {1:2.2f} ".format(cat, customer_features[cat].unique().size))


# In[24]:


#Keep track of customers with missing entry for age
customer_features.age_bucket = customer_features.age_bucket.fillna('Missing_Age') #1565


# In[25]:


#Dummify categorical variables
mlb = MultiLabelBinarizer()
X = customer_features
age_bucket = mlb.fit_transform([{str(val)} for val in X['age_bucket'].values]) 
age_bucket_ = pd.DataFrame(age_bucket, columns=sorted(X['age_bucket'].unique()))  
returning_cust = mlb.fit_transform([{str(val)} for val in X['is_returning_customer'].values])
returning_cust_ = pd.DataFrame(returning_cust, columns =sorted(X['is_returning_customer'].unique())) 
returning_cust_.rename(columns={True: 'Returning', False: 'NonReturning'}, inplace=True) 


# In[26]:


genres = customer_features[['favorite_genres']]
genres['favorite_genres'] = genres.favorite_genres.apply(lambda x: x.strip(' []').split(', '))
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(genres['favorite_genres'])
genre = pd.DataFrame(X, columns=mlb.classes_)

#cleanup
genre = genre[["'Beach-Read'", "'Biography'", "'Classic'", "'Drama'", "'History'", "'Pop-Psychology'", "'Pop-Sci'", "'Romance'",
       "'Sci-Fi'", "'Self-Help'", "'Thriller'"]]
genre.columns = ["Beach-Read", "Biography", "Classic", "Drama","History", "Pop-Psychology", "Pop-Sci", "Romance",
       "Sci-Fi", "Self-Help", "Thriller"]


# In[27]:


cust_features = customer_features[['customer_id']]
cust_features = pd.concat([cust_features,age_bucket_, returning_cust_, genre], axis=1)
#cust_features.head()


# Product Features

# In[28]:


# for cat in ['fiction', 'genre']: 
#     print("Number of levels in category '{0}': \b {1:2.2f} ".format(cat, prod_features[cat].unique().size))


# In[29]:


mlb = MultiLabelBinarizer()
X = prod_features
genre_ = pd.get_dummies(X['genre'], prefix='genre')
fiction = mlb.fit_transform([{str(val)} for val in X['fiction'].values])
fiction_ = pd.DataFrame(fiction, columns = sorted(X['fiction'].unique())) 
fiction_.rename(columns={True: 'Fiction', False: 'NonFiction'}, inplace=True)


# In[30]:


prod_features = prod_features[['product_id', 'length', 'difficulty']]
prod_features = pd.concat([prod_features, genre_,  fiction_], axis=1)
#prod_features.head()


# In[31]:


#Merging datasets 
#Last Month
dfs = [last_month_assortment, last_purchase_order, prod_features] #prod_features
lm_df_final = reduce(lambda left,right: pd.merge(left,right,on='product_id'), dfs)
lm_final = lm_df_final.merge(cust_features, on='customer_id') #cust_features

#Next Month
dfs_nm = [next_month_assortment, last_purchase_order, prod_features]
nm_df_final = reduce(lambda left,right: pd.merge(left,right,on='product_id'), dfs_nm)
nm_final = nm_df_final.merge(cust_features, on='customer_id') 


# Model Selection
# 
# Predict whether a Customer's Purchase is True or False, for the next month.

# In[32]:


lm_final.rename(columns={'purchased': 'Class'}, inplace=True) 
lm_class = lm_final['Class'].values


# In[33]:


lm_features = lm_final.drop('Class', axis = 1)
lm_features = lm_features.values


# In[34]:


nm_final_X = nm_final[['quantity_purchased', 'cost_to_buy','retail_value', 'length', 'difficulty', 'genre_Beach-Read',
       'genre_Biography', 'genre_Classic', 'genre_Drama', 'genre_History','genre_Pop-Psychology', 'genre_Pop-Sci', 'genre_Romance',
       'genre_Sci-Fi', 'genre_Self-Help', 'genre_Thriller', 'NonFiction',
       'Fiction', '0-17', '18-25', '26-35', '36-45', '46-55', '56-65', '66+','Missing_Age', 'NonReturning', 'Returning', 'Beach-Read', 'Biography',
       'Classic', 'Drama', 'History', 'Pop-Psychology', 'Pop-Sci', 'Romance','Sci-Fi', 'Self-Help', 'Thriller']]


# In[35]:


nm_final_X = MinMaxScaler().fit_transform(nm_final_X)


# In[36]:


y = lm_class
X = lm_final[['quantity_purchased', 'cost_to_buy','retail_value', 'length', 'difficulty', 'genre_Beach-Read',
       'genre_Biography', 'genre_Classic', 'genre_Drama', 'genre_History','genre_Pop-Psychology', 'genre_Pop-Sci', 'genre_Romance',
       'genre_Sci-Fi', 'genre_Self-Help', 'genre_Thriller', 'NonFiction',
       'Fiction', '0-17', '18-25', '26-35', '36-45', '46-55', '56-65', '66+','Missing_Age', 'NonReturning', 'Returning', 'Beach-Read', 'Biography',
       'Classic', 'Drama', 'History', 'Pop-Psychology', 'Pop-Sci', 'Romance','Sci-Fi', 'Self-Help', 'Thriller']]


# In[37]:


x = lm_final[['quantity_purchased', 'cost_to_buy','retail_value', 'length', 'difficulty', 'genre_Beach-Read',
       'genre_Biography', 'genre_Classic', 'genre_Drama', 'genre_History','genre_Pop-Psychology', 'genre_Pop-Sci', 'genre_Romance',
       'genre_Sci-Fi', 'genre_Self-Help', 'genre_Thriller', 'NonFiction',
       'Fiction', '0-17', '18-25', '26-35', '36-45', '46-55', '56-65', '66+','Missing_Age', 'NonReturning', 'Returning', 'Beach-Read', 'Biography',
       'Classic', 'Drama', 'History', 'Pop-Psychology', 'Pop-Sci', 'Romance','Sci-Fi', 'Self-Help', 'Thriller']]


# In[38]:



X = MinMaxScaler().fit_transform(X)


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Random Forest

# In[59]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
 # this function computes subset accuracy
print (accuracy_score(y_test, y_pred))
#the count of true negatives is C_{0,0}, false negatives is C_{1,0}, true positives is C_{1,1} and false positives is C_{0,1}.
print (confusion_matrix(y_test, y_pred))
print (classification_report(y_test, y_pred))


# Gradient Boosting

# In[42]:


from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier(max_features=10, n_estimators=500, 
                                 learning_rate=0.05, random_state= 2015)
gbm.fit(X_train, y_train)


# Model Validation

# In[43]:


import sklearn.metrics
print (sklearn.metrics.roc_auc_score(y_train, gbm.predict_proba(X_train)[:,1]))
print (sklearn.metrics.roc_auc_score(y_test, gbm.predict_proba(X_test)[:,1]))

#As a two-class prediction problem, in which the outcomes are labeled either as positive or negative 
#This is equal to the probability that a classifier will rank a randomly chosen 
#positive instance higher than a randomly chosen negative one


# In[60]:


gbm.fit(X_train, y_train)
y_pred = gbm.predict(X_test)
# this function computes subset accuracy
print (accuracy_score(y_test, y_pred))
#the count of true negatives is C_{0,0}, false negatives is C_{1,0}, true positives is C_{1,1} and false positives is C_{0,1}.
print (confusion_matrix(y_test, y_pred)) 
print (classification_report(y_test, y_pred))


# In[44]:


# import matplotlib.pyplot as plt
# import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.style.use('ggplot')
# get_ipython().run_line_magic('matplotlib', 'inline')

# def plot_gbt_learning(gbt):
#     test_score = np.empty(len(gbt.estimators_))
#     train_score = np.empty(len(gbt.estimators_))
#     for i, pred in enumerate(gbt.staged_predict_proba(X_test)):
#          test_score[i] = sklearn.metrics.roc_auc_score(y_test, pred[:,1])
#     for i, pred in enumerate(gbt.staged_predict_proba(X_train)):
#          train_score[i] = sklearn.metrics.roc_auc_score(y_train, pred[:,1])
#     plt.figure(figsize=(8,6))
#     plt.plot(np.arange(gbt.n_estimators) + 1, test_score, label='Test') 
#     plt.plot(np.arange(gbt.n_estimators) + 1, train_score, label='Train')
# plot_gbt_learning(gbm)


# 
# The above plot shows that an AUC score of 0.82 is reached with around 100 estimators for both training(red) and test sets(blue).

# Model Selection  - Summary

# Although both random forest and gradient boosting are ensemblers, random forest decorrelates decision trees, improving upon bagging, building various decision trees on bootstrapped training samples, but splitting internal nodes in a special way.
# 
# Each time a split is considered within the construction of a decision tree, only a random subset of m of the overall p predictors are allowed to be candidates. At every split, a new subset of predictors is randomly selected.The random forest procedure forces the decision tree building process to use different predictors to split at different times. Should a good predictor be left out of consideration for some splits, it still has many chances to be considered in the construction of other splits. We can’t overfit by adding more trees! At each iteration, the classifier is being trained independently.
# 
# For GBM, the boosting procedure works in a similar way, except that the decision trees are generated in a sequential manner: Each tree is generated using information from previously grown trees; the addition of a new tree improves upon the performance of the previous trees.The trees are now dependent upon one another. Gradient Boosting builds on weak classifiers, adding one at a time so that the next one is trained to improve the already trained ensemble.

# In[50]:


results = gbm.predict(nm_final_X)


# In[51]:


submission = results


# In[52]:


# Create the submission file
final = pd.DataFrame({'CustomerId': nm_final['customer_id'], 'ProductID': nm_final['product_id'], 'Pred_Purchase': submission})


# In[53]:


final[final.Pred_Purchase == True].shape


# Cost Analysis

# In[54]:


nm_final['predicted_purchase'] = final.Pred_Purchase
nm_final['predicted_shipping_costs'] = nm_final[['predicted_purchase']].apply(lambda x: shipping_cost(*x), axis=1)
nm_final['predicted_sales'] = nm_final[['predicted_purchase', 'retail_value']].apply(lambda x: sales(*x), axis=1)
print ('Projected Shipping costs: '+ str(round(sum(nm_final['predicted_shipping_costs']),2)))
print ('Projected Sales:' +str(round(sum(nm_final['predicted_sales']),2)))


# In[55]:


lm = lm_final
lm['sales'] = lm[['Class', 'retail_value']].apply(lambda x: sales(*x), axis=1)
lm['Shipping_costs'] = lm[['Class']].apply(lambda x: shipping_cost(*x), axis=1)
print ('Last month shipping costs: ' +'$'+ str(round(sum(lm['Shipping_costs']),2)))
print ('Last month sales: ' + '$'+str(round(sum(lm['sales']),2)))


# In[56]:


nm = nm_final 
next_purch = next_purchase_order
predicted_shipping_costs = round(sum(nm['predicted_shipping_costs']),2)
predicted_sales = sum(nm['predicted_sales'])
last_month_shipping_costs = sum(lm['Shipping_costs'])
last_month_sales = sum(lm['sales'])


# In[57]:


last_purch_ = last_purchase_order.copy()
total_costs(last_purch_)
total_costs(next_purch)
last_purchase_order['total_cost'] = last_purchase_order['quantity_purchased']*last_purchase_order['cost_to_buy']
Loan_Value = sum(last_purchase_order['total_cost'])
print('The total cost of last months loan ' + '$'+ str(Loan_Value))
print('The total cost of next months loan ' + '$' + str(round(sum(next_purch['cost_by_product']),2)))


# Recommendation

# In[58]:


#value of the loan  =  total cost of all the books that we bought
last_month_loan = round(sum(last_purch_['cost_by_product']),2)
next_month_loan = round(sum(next_purch['cost_by_product']),2)
        
ROI(last_month_loan, next_month_loan, last_month_shipping_costs, predicted_shipping_costs, predicted_sales, last_month_sales)

