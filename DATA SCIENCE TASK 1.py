#!/usr/bin/env python
# coding: utf-8

# # Name:- Vaibhav Ramesh Vhalgade    
# # task no :1  
# 
# 

# # DATA SCIENCE AND BUSSINESS ANALYTICS

# # SPARK FOUNDATION

# task :-Prediction using Supervised ML
# 

# Predict the percentage of an student based on the no. of study hours.
#           

# This is a simple linear regression task as it involves just 2 variables.

# ## importing required libraries

# In[111]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# #importing required data from website

# In[6]:


data = pd .read_csv("http://bit.ly/w-data")
print('data importing sucessful')
data


# for accesing first 10 records 

# In[7]:


data.head(10)


# In[28]:


plt.figure(figsize = (15,5))
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage', color = 'red')  
plt.xlabel('Hours Studied', color = 'green', size = 15)  
plt.ylabel('Percentage Score',color = 'green', size = 15) 
plt.legend()
plt.show()


# for 
checking for the null value
# In[10]:


data.isnull == True


# Here is no null value in data 

# so here is no need of updation

# **next step is visulition of data

# In[29]:


## a simple linear regression task as it involves just 2 variables.


# for  regression we need correlation line

# In[54]:


sns.regplot(y = data['Scores'], x = data['Hours'])
plt.title('REGGRESSION GRAPH', size = 20,color = 'red')
plt.xlabel('Hours',size = 15, color = 'green')
plt.ylabel('Scores',size = 15, color = 'green')
plt.show()


# PREPARING THE DATA FOR MORE 

# we are extracting the values of hours in x  and scores in y

# In[57]:


x = data.iloc[: ,:-1].values


# In[160]:


y = data.iloc[:,1].values


# In[161]:


x


# In[60]:


y


# spltion of data into the train and test using sklearn

# In[162]:


from sklearn.model_selection import train_test_split


# In[163]:


x_train,x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# In[164]:


x_train


# In[165]:


x_test


# In[166]:


y_train


# In[167]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(x_train, y_train) 

print("Training complete.")


# In[168]:


# Plotting the regression line
line = regressor.coef_*x+regressor.intercept_

# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, line);
plt.xlabel('score')
plt.ylabel('Hours')
plt.title('TEST DATA')
plt.show()


# #  prediction of score in he study 9.25 hrs a day???

# In[169]:


print(x_test) # Testing data - In Hours
y_pred = regressor.predict(x_test)


# In[170]:


y_test


# In[146]:


y_pred


# In[176]:


df = pd.DataFrame({'Actual': [x_test], 'Predicted':[y_pred]})  
df 


# In[177]:


# You can also test with your own data
hours = 9.25
hours = np.array(hours).reshape(1, -1)
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# ## Evaulation of the algorithm required 

# In[178]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# ## THANK YOU SPARK FOUNDATION  

# In[ ]:




