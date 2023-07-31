#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install ipython


# In[ ]:





# <h3><span style="color:blue">Project Forecasting AMAZON Stock Market using the SVR, LSTM,XGBOOST,LSTM+CNN,MLP Models<h3>

# <h4> Purpose of the research<h4>
# 

# In this study, we will investigate which algorithm of Machine Learning approach (i.e. SVR, LSTM,LSTM+CNN,MLP,XGBOOST), performs more accurate prediction of AMAZON Stock Index stock market. In order to help, investors in the stock markets make the decision in which indexes will invest, and also to give the best fit of the index by modern models.
# 
# In this project, we will do a quantitative study starting with a data analysis going through the modeling, then we will be introduced and use Machine Learning approach (SVR, LSTM,LSTM+CNN,MLP,XGBOOST). Finally, make a forecast by measuring the performance of each model.

# In[4]:


pip install yfinance


# <h4>Importing libraries<h4>

# In[5]:


get_ipython().system('pip install xgboost')


# In[28]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualization
import seaborn as sns
from pylab import rcParams
from matplotlib import pyplot
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import plot_importance, plot_tree
from sklearn.metrics import r2_score
# Warnings Ignore
import warnings
warnings.filterwarnings("ignore")


# In[29]:


import yfinance as yf
import datetime 


# In[30]:


start = datetime.datetime(2019,1,1) 
end = datetime.datetime(2022,12,28) 


# In[31]:


amzn=yf.Ticker("AMZN")


# In[32]:


AMZN=amzn.history(start=start, end=end)
print(AMZN)


# In[33]:


AMZN=AMZN.drop(columns=['Dividends','Stock Splits'])


# In[34]:


AMZN.head(5)


# <h4>Data Analysis<h4>

# In[35]:


AMZN.isnull().sum()


# We notice that we have no missing values!

# In[36]:


AMZN.describe()


# In[37]:


cols_plot = ['Open', 'High', 'Low','Close']
AMZN[cols_plot].plot(subplots=True, legend=True, figsize=(20, 16))
pyplot.show()


# In[38]:


fig, ax = plt.subplots(4, 2, figsize = (15, 13))
sns.boxplot(x= AMZN["Close"], ax = ax[0,0])
sns.distplot(AMZN['Close'], ax = ax[0,1])
sns.boxplot(x= AMZN["Open"], ax = ax[1,0])
sns.distplot(AMZN['Open'], ax = ax[1,1])
sns.boxplot(x= AMZN["High"], ax = ax[2,0])
sns.distplot(AMZN['High'], ax = ax[2,1])
sns.boxplot(x= AMZN["Low"], ax = ax[3,0])
sns.distplot(AMZN['Low'], ax = ax[3,1])
plt.tight_layout()
plt.show()


# Fortunately, we did not find any extreme values in our dataset, which will make our model powerful
# since the extreme values will not impact. Now we visualize the return and their density:

# In[39]:


# plot the daily return percentage
AMZN['Daily Return'] = AMZN['Close'].pct_change()
AMZN['Daily Return'].plot(figsize=(18,8),legend=True,linestyle=':',marker='o')
plt.title('Plot of MSFT Index daily returns')
plt.show()
sns.displot(AMZN['Daily Return'].dropna(),bins=100,color='green', height=6,aspect=2.5)
plt.title('Density of daily return')
plt.show()


# In[40]:


AMZN.drop('Daily Return', inplace=True, axis=1)


# In[41]:


# Just select the variable Close
close = AMZN[["Close"]]
close.head()


# <h4>Plot the time evolution of daily closing AMZN price<h4>

# In[42]:


plt.rcParams['figure.figsize'] = (15, 8) # Increases the Plot Size
close['Close'].plot(color='blue',grid = True)
plt.title('Daily Close Price for AMZN ')
plt.xlabel('Date: Janv. 1th, 2019 - Dec. 28th, 2022')
plt.ylabel('Values')
plt.legend()
plt.show()


# <h4>Data Preprocessing for closing price of AMZN<h4>

# In[43]:


#Set Target Variable
output_var = AMZN[["Close"]]
#Selecting the Features
features = ["Open", "High","Low"]


# <h4>Normalizing our series<h4>

#   xnorm=(xt -min(xt))/(max(xt-min(xt))

# In[44]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(AMZN[features])
feature_transform= pd.DataFrame(columns=features, data=feature_transform,index=AMZN.index)
feature_transform.head()


# <h4>Creating a Training Set and a Test Set for Amzn<h4>

# In[45]:


#Splitting to Training set and Test set
from sklearn.model_selection import TimeSeriesSplit
timesplit= TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(feature_transform):
    X_train, X_test = feature_transform[:len(train_index)],feature_transform[len(train_index): (len(train_index)+len(test_index))]
    y_train, y_test = output_var[:len(train_index)].values.ravel(),output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()


# In[46]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# <h4>Data Preprocessing for LSTM model<h4>

# In[47]:


#Process the data for LSTM
trainX =np.array(X_train)
testX =np.array(X_test)
X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])


# In[48]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# <h4> Modeling
#     
#     
#     
#   
#  <h4>

# <h4> Building a multivariate LSTM model for the prediction of the closing price
# AMZN<h4>

# In[50]:


#Building a simple LSTM Model
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from keras.layers import LSTM
from tensorflow.keras.layers import Dense
from keras.layers import Dense
from keras.layers.core import Dense, Activation, Dropout
from tensorflow.keras.layers import Flatten
lstm = Sequential()
# Input layer
lstm.add(LSTM(220, input_shape=(1, trainX.shape[1]), activation='relu',return_sequences=True))
lstm.add(Dropout(1e-5))
# Hidden layer
lstm.add(LSTM(units = 120,activation='relu'))
lstm.add(Dropout(1e-5))
# Output layer
lstm.add(Dense(1))
lstm.compile(loss='mse', optimizer='adam')


# In[51]:


history=lstm.fit(X_train, y_train, epochs=600, batch_size=5,validation_split=.75, verbose=1, shuffle=False)


# In[ ]:


lstm.summary()


# <h4>3.2 Diagnosing an LSTM Model<h4>

# In[ ]:


# Plot training & validation loss values
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'],color = 'blue',label="Training loss")
plt.plot(history.history['val_loss'],color = 'red',label="Validation loss")
plt.title('Training and Validation Loss', fontsize=20)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training loss', 'Validation loss'], loc='upper left', fontsize=16)
plt.show()


# • If a model has a low train accuracy and a high train loss, then the model is suffering from
# underfitting.
# 
# • If a model has a high train accuracy but a low validation accuracy then the model is suffering
# from overfitting.
# 
# • If the train accuracy model graph converges or concedes with the validation accuracy then
# the model is optimal.
# 
# ==> In our case we found a good LSTM model to predict the closing price values AMZN.

# In[ ]:


y_pred_test= lstm.predict(X_test)
y_pred_test = y_pred_test.flatten()
plt.plot(y_test, label='True Closing price Value')
plt.plot(y_pred_test, label='LSTM Closing Price Value')
plt.title('Prediction the closing price using LSTM (Test set)',fontsize=16 )
plt.xlabel("Time Scale")
plt.ylabel("Scaled USD")
plt.legend()
plt.show()


# In[ ]:


y_pred_train= lstm.predict(X_train)
y_pred_train = y_pred_train.flatten()
plt.plot(y_train, label='True Closing price Value')
plt.plot(y_pred_train, label='LSTM Closing Price Value')
plt.title('Prediction the closing price using LSTM (Train set)',fontsize=16)
plt.xlabel("Time Scale")
plt.ylabel("Scaled USD")
plt.legend()
plt.show()


# In[ ]:


forecast_accuracy(y_pred_test,y_test)


# In[ ]:


forecast_accuracy(y_pred_train,y_train)


# <h4>Splitting Data<h4>

# Normalisation is not needed for XGBoost model  Since XGBoost is essentially an ensemble algorithm composed of decision trees, it does not require normalization for the inputs. This is because decision trees do not require normalization of their inputs.

# In[ ]:


#Splitting to Training set and Test set
from sklearn.model_selection import TimeSeriesSplit
timesplit= TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(feature_transform):
    X_train, X_test = feature_transform[:len(train_index)],feature_transform[len(train_index): (len(train_index)+len(test_index))]
    y_train, y_test = output_var[:len(train_index)].values.ravel(),output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()


# In[ ]:


X_train =np.array(X_train)
X_test =np.array(X_test)


# In[ ]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# <h4>Build XGBoost Model<h4>

# To create our model we will initialize it with the following hyperparameters:
# 
# • base_score = 0.5
# 
# • booster = ‘gbtree’
# 
# • n_estimators = 1000
# 
# • objective = ‘reg:linear’
# 
# • max_depth = 3
# 
# • learning_rate = 0.01

# The useful technique for developing supervised regression models is XGBoost.The objective function contains loss function and a regularization term. It provides information on
# the difference between actual and predicted values, and how far the model’s predictions depart from
# the actual values.
model_xgboost = xgb.XGBRegressor(base_score=0.5,
booster='gbtree',
n_estimators=1000,
objective='reg:linear',
max_depth=3,
learning_rate=0.01 )
# In[ ]:


#fitting our model
model_xgboost.fit(X_train, y_train,

eval_set=[(X_train, y_train), (X_test, y_test)],
early_stopping_rounds=50,
verbose=False) # Change verbose to True if you want to see it␣train


# In[ ]:


xgb_close=model_xgboost.fit(X_train, y_train)


# In[ ]:


#Show the model Svr prediction
xgb_prediction_close_test = xgb_close.predict(X_test)
xgb_prediction_close_train = xgb_close.predict(X_train)
#print(Svr_prediction_return)


# In[ ]:


#Plot the models on a graph to see which has the best fit to the original data
plt.figure(figsize=(16,8))
plt.plot(y_test,color= 'green', label= 'True Closing price Value ') # plotting the␣dataset 'test'
plt.plot(xgb_prediction_close_test,color= 'red', label= 'XGBOOST Closing Price Value')
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.title('Prediction the closing price using XGBOOST (Test set)', fontsize=16)
plt.legend(['True Closing price Value','XGBOOST Closing Price Value'], fontsize=12)
plt.show()


# In[ ]:


#Plot the models on a graph to see which has the best fit to the original data
plt.figure(figsize=(16,8))
plt.plot(y_train,color= 'green', label= 'True Closing price Value ') # plotting the␣dataset 'test'
plt.plot(xgb_prediction_close_train,color= 'red', label= 'XGBoost Closing Price Value')
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.title('Prediction the closing price using XGBOOST (Train set)', fontsize=16)
plt.legend(['True Closing price Value','XGBOOST Closing Price Value'], fontsize=12)
plt.show()


# In[ ]:


forecast_accuracy(xgb_prediction_close_test,y_test)


# In[ ]:


forecast_accuracy(xgb_prediction_close_train,y_train)


# <h4>Building a SVR Model for closing price of AMZN<h4>

# In[ ]:


from sklearn.svm import LinearSVR
svm_reg_close = LinearSVR(epsilon=5e-4)


# In[ ]:


reg_close=svm_reg_close.fit(X_train,y_train)


# In[ ]:


#Show the model Svr prediction
Svr_prediction_close_test = reg_close.predict(X_test)
Svr_prediction_close_train = reg_close.predict(X_train)
#print(Svr_prediction_return)


# In[ ]:


#Plot the models on a graph to see which has the best fit to the original data
plt.figure(figsize=(16,8))
plt.plot(y_test,color= 'blue', label= 'True Closing price Value ') # plotting the␣dataset 'test'
plt.plot(Svr_prediction_close_test,color= 'yellow', label= 'LinearSVR Closing Price Value')
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.title('Prediction the closing price using SVR (Test set)', fontsize=16)
plt.legend(['True Closing price Value','LinearSVR Closing Price Value'], fontsize=12)
plt.show()


# In[ ]:


#Plot the models on a graph to see which has the best fit to the original data
plt.figure(figsize=(16,8))
plt.plot(y_train,color= 'blue', label= 'True Closing price Value ') # plotting the␣dataset 'test'
plt.plot(Svr_prediction_close_train,color= 'yellow', label= 'LinearSVR Closing Price Value')
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.title('Prediction the closing price using SVR (Test set)', fontsize=16)
plt.legend(['True Closing price Value','LinearSVR Closing Price Value'], fontsize=12)
plt.show()


# In[ ]:


forecast_accuracy(Svr_prediction_close_train,y_train)


# In[ ]:


forecast_accuracy(Svr_prediction_close_test,y_test)


# <h4>Building MLP Model<h4>

# In[ ]:


from sklearn.neural_network import MLPRegressor
# init a model
model_MLP_skl = MLPRegressor(hidden_layer_sizes=(50), activation='relu', solver= 'sgd',

alpha = 0.0001,
batch_size=32,
learning_rate_init=0.001,
max_iter = 1000, verbose = False,
early_stopping=True)


# In[ ]:


model_MLP_skl.fit(X_train, y_train )


# In[ ]:


ml_test = model_MLP_skl.predict(X_test)
ml_train=model_MLP_skl.predict(X_train) 


# In[ ]:


#Plot the models on a graph to see which has the best fit to the original data
plt.figure(figsize=(16,8))
plt.plot(y_test,color= 'blue', label= 'True Closing price Value ') # plotting the␣dataset 'test'
plt.plot(ml_test,color= 'Orange', label= 'MLP Closing Price Value')
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.title('Prediction the closing price using MLP (Test set)', fontsize=16)
plt.legend(['True Closing price Value','MLP Closing Price Value'], fontsize=12)
plt.show()


# In[ ]:


#Plot the models on a graph to see which has the best fit to the original data
plt.figure(figsize=(16,8))
plt.plot(y_train,color= 'blue', label= 'True Closing price Value ') # plotting the␣dataset 'test'
plt.plot(ml_train,color= 'Orange', label= 'MLP Closing Price Value')
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.title('Prediction the closing price using MLP (Train set)', fontsize=16)
plt.legend(['True Closing price Value','MLP Closing Price Value'], fontsize=12)
plt.show()


# In[ ]:


forecast_accuracy(ml_train,y_train)


# In[ ]:


forecast_accuracy(ml_test,y_test)


# <h4>Data Preparation<h4>

# Before a multivariate series can be modeled, it must be prepared. The CNN model will learn a
# function that maps a sequence of past observations as input (X) to observation as output (Y). To
# make our model more predictive, we have vectorized the input data, which means that when we
# train the model in a sequence of 50 several numbers of data will be introduced in the model 

# In[ ]:


from sklearn.model_selection import train_test_split
# split a multivariate sequence into samples
X = []
Y = []
# define input sequence
window_size=50 # choose a number of time steps
for i in range(1 , len(AMZN) - window_size -1 , 1):
    first = AMZN.iloc[i,2]
    temp = []
    temp2 = []
    for j in range(window_size): # gather input and output parts of the pattern
        temp.append((AMZN.iloc[i + j, 2] - first) / first)
    temp2.append((AMZN.iloc[i + window_size, 2] - first) / first)
    X.append(np.array(temp).reshape(50, 1))
    Y.append(np.array(temp2).reshape(1, 1))
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,shuffle=True) # convert into input/output
# define input and output sequences
train_X = np.array(x_train)
test_X = np.array(x_test)
train_Y = np.array(y_train)
test_Y = np.array(y_test)
# convert to [rows, columns] structure, the dataset knows the number of␣features, e.g. 0 (i.e. train_X.shape[0])
train_X = train_X.reshape(train_X.shape[0],1,50,1)
test_X = test_X.reshape(test_X.shape[0],1,50,1)
print(len(train_X))
print(len(test_X))


# In[ ]:


print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)


#  1D CNN
# model expects data to have the shape of: [samples, timesteps, features]. But in our case we
# will work with CNN-LSTM model i.e. we’ll reshape input from [samples, timesteps, features]
# into [samples, subsequences, timesteps, features].

# <h4>Building CNN+LSTM MODEL<h4>

# To build the CNN model we will first initialize the sequential and then add each layer (Conv1D,
# MaxPooling1D, Conv1D, MaxPooling1D,Conv1D, MaxPooling1D, Flatten). The Rectified Linear
# Unit (ReLU) function is the most simple and most used activation function. The ReLU function
# can be defined as followings:ReLU(X) = max(0, x)
# 
# Then we add the LSTM model layers (Bidirectional, Dropout, Bidirectional, Dropout). The process
# can be summarized as follows:
# Using a 1D convolution with three layers followed by a maximum pooling layer, the layers are
# created with sizes 64,128,64 with kernel size = 3. The output is then flattened to feed the Bi-LSTM
# layers. The model has two hidden LSTM layers followed by a dense layer to provide the output.

# <h4>importing librairies<h4>

# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional,TimeDistributed
from tensorflow.keras.layers import MaxPooling1D, Flatten
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.metrics import RootMeanSquaredError


# In[ ]:


# define model CNN layers
model = tf.keras.Sequential()

# first input model CNN_1 with three layers
model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu',input_shape=(None, 49, 1))))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(128, kernel_size=3, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Flatten()))
# model.add(Dense(5, kernel_regularizer=L2(0.01)))
# LSTM layers
model.add(Bidirectional(LSTM(50, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(50, return_sequences=False)))
model.add(Dropout(0.5))
#Final layers
model.add(Dense(1, activation='linear'))


# The Adam (Adaptive Moment Estimation) optimizer, one of the most common optimization tech-
# niques, is used to compile the model after it has been generated:

# In[ ]:


# Compile the Model
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])


# In[ ]:


# Fit CNN-LSTM model 'Train the Model'
history = model.fit(train_X, train_Y, validation_data=(test_X,test_Y),epochs=60,batch_size=40, verbose=1, shuffle =True)


# In[ ]:


# demonstrate the model parameters
model.summary()


# In[ ]:


# After the model has been constructed, we'll summarise it
from tensorflow.keras.utils import plot_model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


# In[ ]:


model.evaluate(test_X, test_Y)


# In[ ]:


test_eval = model.evaluate(test_X, test_Y, verbose=0)


# In[ ]:


print('Loss:', test_eval[0])
print('MSE:', test_eval[1])
print('MAE:', test_eval[2])


# <h4>Forecasting<h4>

# In[ ]:


from sklearn.metrics import explained_variance_score, mean_poisson_deviance,mean_gamma_deviance
from sklearn.metrics import r2_score
from sklearn.metrics import max_error
# predict probabilities for test set
yhat_probs = model.predict(test_X, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
var = explained_variance_score(test_Y.reshape(-1,1), yhat_probs)
print('Variance: %f' % var)
r2 = r2_score(test_Y.reshape(-1,1), yhat_probs)
print('R2 Score: %f' % var)
var2 = max_error(test_Y.reshape(-1,1), yhat_probs)
print('Max Error: %f' % var2)


# We can see that our R2 = 84% close to 1 which is good, it will allow us to see a good
# forecast in the test.

# In[ ]:


predicted_test = model.predict(test_X)
test_label = test_Y.reshape(-1,1)
predicted_test = np.array(predicted_test[:,0]).reshape(-1,1)
len_t = len(train_X)
for j in range(len_t , len_t + len(test_X)):
    temp = AMZN.iloc[j,3]
    test_label[j - len_t] = test_label[j - len_t] * temp + temp
    predicted_test[j - len_t] = predicted_test[j - len_t] * temp + temp
fig = plt.figure(figsize=(15,5))
plt.plot(predicted_test, color = 'green', label = 'Predicted Stock Price')
plt.plot(test_label, color = 'red', label = 'Real Stock Price')
plt.title(' Stock Price Prediction (test set)')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.show()


# In[ ]:


forecast_accuracy(predicted_test,test_label)


# <h4> Benchmarking <h4>

# <table style='float:left;'>
#     <tr>
#         <th style='text-allign: center;'>Errors</th>
#         <th style='text-allign: center;'>Series</th>
#         <th style='text-allign: center;'>ME</th>
#         <th style='text-allign: center;'>MAPE(%)</th>
#         <th style='text-allign: center;'>MAE</th>
#         <th style='text-allign: center;'>MPE</th>
#         <th style='text-allign: center;'>RMSE</th>
#     </tr>
#     <tr>
#         <td style='text-allign: center;'>LSTM</td>
#         <td style='text-allign: center;'>Train</td>
#         <td style='text-allign: center;'>3.57</td>
#         <td style='text-allign: center;'>2.41</td>
#         <td style='text-allign: center;'>3.76</td>
#         <td style='text-allign: center;'>0.02</td>
#         <td style='text-allign: center;'>5.13</td>
#       </tr> 
#     <tr>
#         <td style='text-allign: center;'>LSTM</td>
#         <td style='text-allign: center;'>test</td>
#         <td style='text-allign: center;'>0.61</td>
#         <td style='text-allign: center;'>0.48</td>
#         <td style='text-allign: center;'> 0.64</td>
#         <td style='text-allign: center;'>6.28e-05</td>
#         <td style='text-allign: center;'>0.87</td>
#        </tr>
#     <tr>
#         <td style='text-allign: center;'>XGBOOST</td>
#         <td style='text-allign: center;'>Train</td>
#         <td style='text-allign: center;'>-0.005</td>
#         <td style='text-allign: center;'>0.48</td>
#         <td style='text-allign: center;'>0.64</td>
#         <td style='text-allign: center;'>6.28e-05</td>
#         <td style='text-allign: center;'>0.87</td>
#     </tr> 
#     <tr>
#         <td style='text-allign: center;'>XGBOOST</td>
#         <td style='text-allign: center;'>Test</td>
#         <td style='text-allign: center;'>0.23</td>
#         <td style='text-allign: center;'>1.12</td>
#         <td style='text-allign: center;'>1.21</td>
#         <td style='text-allign: center;'>0.002</td>
#         <td style='text-allign: center;'>1.60</td>
#         </tr> 
#     <tr>
#         <td style='text-allign: center;'>SVR</td>
#         <td style='text-allign: center;'>Train</td>
#         <td style='text-allign: center;'>-0.18</td>
#         <td style='text-allign: center;'>0.85</td>
#         <td style='text-allign: center;'>1.14</td>
#         <td style='text-allign: center;'>-0.001</td>
#         <td style='text-allign: center;'>1.53</td>
#         </tr> 
#     <tr>
#         <td style='text-allign: center;'>SVR</td>
#         <td style='text-allign: center;'>Test</td>
#         <td style='text-allign: center;'> 0.13</td>
#         <td style='text-allign: center;'> 1.20</td>
#         <td style='text-allign: center;'> 1.30</td>
#         <td style='text-allign: center;'>0.0016</td>
#         <td style='text-allign: center;'> 1.67</td>
#         </tr> 
#     <tr>
#         <td style='text-allign: center;'>MLP</td>
#         <td style='text-allign: center;'>Train</td>
#         <td style='text-allign: center;'>0.039</td>
#         <td style='text-allign: center;'>0.85</td>
#         <td style='text-allign: center;'>1.14</td>
#         <td style='text-allign: center;'>0.0004</td>
#         <td style='text-allign: center;'>1.53</td>
#         </tr> 
#     <tr>
#         <td style='text-allign: center;'>MLP</td>
#         <td style='text-allign: center;'>Test</td>
#         <td style='text-allign: center;'> 0.34</td>
#         <td style='text-allign: center;'>1.25</td>
#         <td style='text-allign: center;'>1.35</td>
#         <td style='text-allign: center;'> 0.0036</td>
#         <td style='text-allign: center;'>1.71</td>
#         </tr> 
#       <tr>
#         <td style='text-allign: center;'>CNN+LSTM</td>
#         <td style='text-allign: center;'>Test</td>
#         <td style='text-allign: center;'>-1.00</td>
#         <td style='text-allign: center;'>3.63</td>
#         <td style='text-allign: center;'>4.42</td>
#         <td style='text-allign: center;'>-0.005</td>
#         <td style='text-allign: center;'>6.62</td>
#         </tr> 
#    
#     
#     
#    
#         
#     

# <h4> Conclusion <h4>

# The conclusions that can be extracted from this study are the following:
# • First of all, all models need an initiative to improve them better.
# 
# • The Results from the metrics  of models (SVR,MLP,XGBOOST) are really close 
# 
# • The results of CNN+LSTM model is not enough tries to find good hyperparamterer that generate
# well the training and test data.
# 
# • Try to improve the results of the LSTM model (i.e. change the hyper parameters of the model)
# performance measurements until you find the train set errors less than the test set.
# 
# • Scaling the original and predicted values of the training and test sets in the LSTM model,
# and plot the graphs of the predicted training and test values, then calculate the performance
# measures.
# 
#  

# In[ ]:




