#!/usr/bin/env python
# coding: utf-8

# # Demand Forecasting Problem

# In[1]:


import numpy as np
import pylab as pl
import pandas as pd
import time


# ## Data Preparation

# In[2]:


#Importing the dataset
#Demand = pd.read_excel("C:\Users\weimar\Desktop\DB_2.xlsx", sheet_name='Clean Data')
Demand = pd.read_excel(r'C:\Users\weima\OneDrive\Escritorio\DB_2.xlsx', sheet_name='Clean Data')
data = pd.DataFrame(Demand)


# In[3]:


##Dataset visualization
data


# ### Features

# In[4]:


feature_cols = [ 'Month', 'Whse_A', 'Whse_C', 'Whse_J', 'Whse_S']


# ### Categorical Data

# In[5]:


#Binary variables creation
Warehouse = data.Warehouse
D_Warehouse = pd.get_dummies(Warehouse)


# In[6]:


#Creating the new dataset
Data_frame = pd.concat([data, D_Warehouse], axis=1, join_axes=[D_Warehouse.index])

#Data visualization sample
Data_frame [:11]


# In[8]:


Data_frame.to_excel("output.xlsx")


# In[8]:


#Product_List = pd.read_excel("C:\Users\weimar\Desktop\DB_2.xlsx", sheet_name='Product List')
Product_List = pd.read_excel(r'C:\Users\weima\OneDrive\Escritorio\DB_2.xlsx', sheet_name='Product List')
#Product list visualization sample
Product_List [:11]


# ## Variables Definition

# In[9]:


from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# ## Linear Regression

# In[17]:


#For this problem, every product need a prediction about its future behavior
#For each product is necessary to extract information

LR_score_means = []
LR_MAD = []
LR_MAPE= []

t = time.clock()

for i in range (0, 164):
 Product=Data_frame.loc[Data_frame.Product_Code==Product_List.Product_Code[i],:]
 #Predictors definition
 X = Product[feature_cols] 

 #Target definition
 Y = Product.Order_Demand 

 #Data Splitting
 X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
 
 #Linear Regression
 Linear_Reg = LinearRegression()
 Linear_Reg.fit(X_train, Y_train)
 Linear_pred = Linear_Reg.predict(X_test)

 #Linear Regression Evaluation
 LR_scores = cross_val_score(Linear_Reg, X_train, Y_train, cv=10)
 LR_score_means.append(LR_scores.mean())
    
 Linear_Reg_MAD = np.sum(abs(Linear_pred - Y_test)/len(Y_test))
 LR_MAD.append(Linear_Reg_MAD)

 Linear_Reg_MAPE = np.sum(abs(Linear_pred - Y_test)/Y_test)
 LR_MAPE.append(Linear_Reg_MAPE)

#Time counter 
if (time.clock() - t) < 60 :
 print "%.2f sec" % (time.clock() - t)
else:
 print "%.2f min" % ((time.clock() - t)/60)


# ## KernelRidge Regression

# In[18]:


#Parameters for KR Regression

parameters = [{'kernel': ['rbf'],'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],'alpha': [0.1, 0.01, 0.001]}, {'kernel': ['poly'],'degree': [3],
              'alpha': [0.1, 0.01, 0.001]}, {'kernel': ['linear'],'alpha': [0.1, 0.01, 0.001]}]


# In[19]:


#KR Regression
KR_score_mean = []
KR_MAD = []
KR_MAPE= []

t = time.clock()

for i in range (0, 164):
 Product=Data_frame.loc[Data_frame.Product_Code==Product_List.Product_Code[i],:]
 #Predictors definition
 X1 = Product[feature_cols] 
 #Target definition
 Y1 = Product.Order_Demand 

 #Data Splitting
 X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size=0.25, random_state=0)

 #Kernel-Ridge Regression
 Kernel_Ridge = KernelRidge()
 KR_Selection = GridSearchCV(Kernel_Ridge,parameters, cv=10)
 
 KR_Selection.fit(X_train, Y_train)
 KR_pred = KR_Selection.predict(X_test)
 
 #KR Regression Evaluation
 KR_score = cross_val_score(KR_Selection, X_train, Y_train, cv=10)
 KR_score_mean.append(KR_score.mean())

 KR_Regression_MAD = np.sum(abs(KR_pred - Y_test)/len(Y))
 KR_MAD.append(KR_Regression_MAD)

 KR_Regression_MAPE = (np.sum(abs(KR_pred - Y_test)/Y_test))/len(Y)
 KR_MAPE.append(KR_Regression_MAPE)

#Time counter 
if (time.clock() - t) < 60 :
 print "%.2f sec" % (time.clock() - t)
else:
 print "%.2f min" % ((time.clock() - t)/60)


# ## Neural Network

# In[24]:


nn_parameters = [{'activation': ['logistic'],'alpha': [0.0001, 0.00001], 'learning_rate':['adaptive']}, {'activation': ['relu'],'alpha': [0.0001, 0.00001], 'learning_rate':['adaptive']} ]
MLP_15 = MLPRegressor(hidden_layer_sizes=(15, ), solver='adam', batch_size='auto', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


NN_MLP_15_score =[]
NN_MLP_15_MAD = []
NN_MLP_15_MAPE = []

import time
t = time.clock()

for i in range (0, 164):
 Product=Data_frame.loc[Data_frame.Product_Code==Product_List.Product_Code[i],:]
 #Predictors definition
 X = Product[feature_cols] 

 #Target definition
 Y = Product.Order_Demand 

 #Data Splitting
 X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

 #Neuronal Network
 MLP_15_Selection = GridSearchCV(MLP_15, nn_parameters, cv=10)
 MLP_15_Selection.fit(X_train, Y_train)
 MLP_15_pred = MLP_15_Selection.predict(X_test)

 #KR Regression Evaluation
 MLP_15_score = MLP_15_Selection.score(X_train, Y_train)
 NN_MLP_15_score.append(MLP_15_score)

 MLP_15_MAD = np.sum(abs(MLP_15_pred - Y_test)/len(Y))
 NN_MLP_15_MAD.append(MLP_15_MAD)

 MLP_15_MAPE = (np.sum(abs(MLP_15_pred - Y_test)/Y_test))/len(Y)
 NN_MLP_15_MAPE.append(MLP_15_MAPE)

#Time counter 
if (time.clock() - t) < 60 :
 print "%.2f sec" % (time.clock() - t)
else:
 print "%.2f min" % ((time.clock() - t)/60)


# In[26]:


nn_parameters = [{'activation': ['logistic'],'alpha': [0.0001, 0.00001], 'learning_rate':['adaptive']}, {'activation': ['relu'],'alpha': [0.0001, 0.00001], 'learning_rate':['adaptive']} ]
MLP_10 = MLPRegressor(hidden_layer_sizes=(10, ), solver='adam', batch_size='auto', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


NN_MLP_10_score =[]
NN_MLP_10_MAD = []
NN_MLP_10_MAPE = []

import time
t = time.clock()

for i in range (0, 164):
 Product=Data_frame.loc[Data_frame.Product_Code==Product_List.Product_Code[i],:]
 #Predictors definition
 X = Product[feature_cols] 

 #Target definition
 Y = Product.Order_Demand 

 #Data Splitting
 X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

 #Neuronal Network
 MLP_10_Selection = GridSearchCV(MLP_10, nn_parameters, cv=10)
 MLP_10_Selection.fit(X_train, Y_train)
 MLP_10_pred = MLP_10_Selection.predict(X_test)

 #KR Regression Evaluation
 MLP_10_score = MLP_10_Selection.score(X_train, Y_train)
 NN_MLP_10_score.append(MLP_10_score)

 MLP_10_MAD = np.sum(abs(MLP_10_pred - Y_test)/len(Y))
 NN_MLP_10_MAD.append(MLP_10_MAD)

 MLP_10_MAPE = (np.sum(abs(MLP_10_pred - Y_test)/Y_test))/len(Y)
 NN_MLP_10_MAPE.append(MLP_10_MAPE)

#Time counter 
if (time.clock() - t) < 60 :
 print "%.2f sec" % (time.clock() - t)
else:
 print "%.2f min" % ((time.clock() - t)/60)


# # Results

# In[27]:


Product_code = []
for i in range (0,164):
 Product_code.append(Product_List.Product_Code[i])


# In[30]:


Score = {'(01)-Product_Code': Product_code,'(02)-Linear Regression': LR_score_means, '(03)-KR Regression': KR_score_mean, '(04)-Neural Network (15)':NN_MLP_15_score, '(05)-Neural Network (10)':NN_MLP_10_score}
Score_data_frame = pd.DataFrame(data=Score)
Score_data_frame


# In[31]:


MAD = {'Product_Code': Product_code,'Linear Regression': LR_MAD, 'KR Regression': KR_MAD, 'Neural Network(15)':NN_MLP_15_MAD, 'Neural Network(10)':NN_MLP_10_MAD }
MAD_data_frame = pd.DataFrame(data=MAD)
MAD_data_frame


# In[32]:


MAPE = {'Product_Code': Product_code, 'Linear Regression': LR_MAPE, 'KR Regression': KR_MAPE, 'Neural Network (15)':NN_MLP_15_MAPE, 'Neural Network (10)': NN_MLP_10_MAPE}
MAPE_data_frame = pd.DataFrame(data=MAPE)
MAPE_data_frame[135:145]


# # Model Selection

# In[37]:


Selector = []
MAPE = []
for i in range (0, 164):
 if min (NN_MLP_15_MAPE[i], NN_MLP_10_MAPE[i], KR_MAPE[i], LR_MAPE[i])== LR_MAPE[i]:
        Selector.append("Linear Regression")
        MAPE.append(LR_MAPE[i])
 if min (NN_MLP_15_MAPE[i], NN_MLP_10_MAPE[i], KR_MAPE[i], LR_MAPE[i])== KR_MAPE[i]:
        Selector.append("Kernel Ridge Regression")
        MAPE.append(KR_MAPE[i])
 if min (NN_MLP_15_MAPE[i], NN_MLP_10_MAPE[i], KR_MAPE[i], LR_MAPE[i])== NN_MLP_15_MAPE[i]:
        Selector.append("MLP Neural Network 15")
        MAPE.append(NN_MLP_15_MAPE[i])
 if min (NN_MLP_15_MAPE[i], NN_MLP_10_MAPE[i], KR_MAPE[i], LR_MAPE[i])== NN_MLP_10_MAPE[i]:
        Selector.append("MLP Neural Network 10")
        MAPE.append(NN_MLP_10_MAPE[i])


# In[38]:


Results = {'Product_Code': Product_code, 'Model_Selection': Selector}
Results_data_frame = pd.DataFrame(data=Results)
Results_data_frame [135:145]


# In[40]:


a = 0 #Number of products that fit to a linear regression
b = 0 #Number of products that fit to a KR regression
c = 0 #Number of products that fit to the MLP_15 Neural Network
d = 0 #Number of products that fit to the MLP_10 Neural Network
for i in range (0, 164):
 if Results_data_frame.Model_Selection[i]=="Linear Regression":
    a = a + 1
 if Results_data_frame.Model_Selection[i]=="Kernel Ridge Regression":
    b = b + 1
 if Results_data_frame.Model_Selection[i]=="MLP Neural Network 15":
    c = c + 1
 if Results_data_frame.Model_Selection[i]=="MLP Neural Network 10":
    d = d + 1
print a
print b
print c
print d


# In[41]:


LE = []
PL_LE = []
E = []
for i in range (0,164):
 if MAPE[i] <0.2:
  LE.append(Results_data_frame.Model_Selection[i])  
  PL_LE.append(Results_data_frame.Product_Code[i])
  E.append(MAPE[i])   


# In[42]:


Results2 = {'Product_Code': PL_LE, 'Model_Selection': LE, 'Error (%)': E}
Results2_data_frame = pd.DataFrame(data=Results2)
Results2_data_frame


# ## Predictions

# In[44]:


MLP_15_pred[:10]


# In[45]:


MLP_10_pred[:10]


# In[46]:


Linear_pred[:10]


# In[47]:


KR_pred[:10]


# In[ ]:




