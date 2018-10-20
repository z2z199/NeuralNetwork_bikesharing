# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:42:58 2018

@author: zhuli20
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 09:23:12 2018

@author: zhuli20
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('F:/Bike_sharing/Bike-Sharing-Dataset/day.csv')

#Simple statistic description
Sta_plot_data=data[:300]
Sta_plot_data.plot(x='dteday',y='cnt')
plt.show()


#Data_prep
#Make dummy varibales for season,weekday,weathersit

var=['mnth','season','weekday','weathersit']
for dummy_var in var:
    dummies=pd.get_dummies(data[dummy_var],prefix=dummy_var,drop_first=False)
    data = pd.concat([data, dummies], axis=1)

#Concat DATA

data=data.drop(var,axis=1)

#Standarize features
       
features=['temp','atemp','hum','windspeed','cnt']
for var in features:
    mean,std=data[var].mean(),data[var].std()
    data.loc[:,var]=(data[var]-mean)/std

#Split 0ff last 30% of data for testing
#Choose the first 512 days of the data
train_data=data[:512]
#The rest 30% for test
test_data=data[512:]

#Split into features and targets
drop=['instant','dteday','casual','registered','cnt']
train_features,train_targets=train_data.drop(drop,axis=1),train_data[['cnt']]
test_features,test_targets=test_data.drop(drop,axis=1),test_data[['cnt']]

#Create my NeuralNwtwork class
class NeuralNetwork(object):
    def __init__(self,n_input,n_hidden,n_output,learnrate,n_records):
       #Set numbers of  hidden output
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.n_output=n_output
        self.learnrate=learnrate
        self.n_records=n_records

#Initialize weights
        self.weights_input_to_hidden=np.random.normal(0,scale=self.n_input**-0.5,size=(self.n_input,self.n_hidden))
        self.weights_hidden_to_output=np.random.normal(0,scale=self.n_hidden**-0.5,size=(self.n_hidden,self.n_output))

#Def sigmoid
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

#Forward pass
    #Def train model to update weights
    def train_model(self,input_values,output_values):

               #Def empty input output matrix
               input=np.array(input_values,ndmin=2)
               output=np.array(output_values,ndmin=2)
               # Calculate the hidden layer output
               hidden_in=np.dot(input,self.weights_input_to_hidden)#input(512,10),w_i_h(10,12)  hid(512,12)
               hidden_out=self.sigmoid(hidden_in) #shape(512,12)
               #Calculate the output layer
               output_out=np.dot(hidden_out,self.weights_hidden_to_output)#(512,1)

#Backward pass
               #Calculate output error term
               output_error=output-output_out #shape(512,1)
               output_error_term=output_error

#Calculate hidden error term
               hidden_error=np.dot(output_error,self.weights_hidden_to_output.T) #shape(512,12)
               hidden_error_term=hidden_error*hidden_out*(1-hidden_out) #shape(512,12)

#Calculate delta_w and Update it
               #Def empty delta_w
               delta_w_hidden_to_output=np.zeros(self.weights_hidden_to_output.shape)
               delta_w_input_to_hidden=np.zeros(self.weights_input_to_hidden.shape)
               #Calculate and update
               delta_w_hidden_to_output+=np.dot(hidden_out.T,output_error_term)#shape(12,1)
               delta_w_input_to_hidden+=np.dot(hidden_error_term.T,input).T #shape(12,10)

               #Change w
               self.weights_input_to_hidden+=self.learnrate*delta_w_input_to_hidden/self.n_records
               self.weights_hidden_to_output+=self.learnrate*delta_w_hidden_to_output/self.n_records

    #Def function run for output
    def run(self,features):
        input_in=np.dot(features,self.weights_input_to_hidden)
        input_out=self.sigmoid(input_in)
        out_out=np.dot(input_out,self.weights_hidden_to_output)
        return out_out

    #Def function for MSE
    def MSE(self,output,targets):
        return np.mean((output-targets)**2)

#TRAIN DATA
#Set numbers of hidden layer,epochs,learnrate
n_hidden=14
n_output=1
n_input=train_features.shape[1]
learnrate=0.8
n_records=train_features.shape[0]
epochs=2000
net=NeuralNetwork(n_input,n_hidden,n_output,learnrate,n_records)

for i in range(epochs):
    input,output=train_features.values,train_targets.values
    train=net.train_model(input,output)
    output_train=net.run(input)
    #Calculate MSE in train data
    MSE_train=net.MSE(output_train,output)
    print(MSE_train)


#TEST MODEL
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(10,5))
mean,std=np.mean(data['cnt']),np.std(data['cnt'])
pre_cnt=net.run(test_features)*std+mean
plot_test_data=test_targets.values*std+mean
print(pre_cnt.shape)
print(plot_test_data.shape)
#plot
ax.plot(pre_cnt,label='prediction')
ax.plot(plot_test_data,label='targets')
ax.set_xlim(right=len(pre_cnt))
ax.legend()
#Set axis
dates=pd.to_datetime(data.ix[test_data.index]['dteday'])
dates=dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks=(np.arange(len(dates))[1::1])
_=ax.set_xticklabels(dates[1::1],rotation=45)
#Show pic
plt.show()











































