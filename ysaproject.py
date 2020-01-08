# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 03:59:31 2019

@author: uasmt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
# filter warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('pulsar_stars.csv')

#dat özellikleri verilen yıldızın , pulsar yıldızı olup olmadıgını anlayacak
# verileri x, input ve y , output olarak ayırıyoruz
y = data["target_class"].values.reshape(-1,1)
x = data.drop(["target_class"],axis=1)

#datayı eğitmek ve test için sklearn kutuphanesini kullanarak %80 train,%20 test olarak ayırıyoruz
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# elde ettiğimiz verilerin transposunu alıyoruz
# backward ve predict işlemleri yaparken matrix hatasını gidermek için bunu yaptık, forward ı buna göre yeniledik 
x_train = X_train.T
y_train = Y_train.T
x_test = X_test.T
y_test = Y_test.T

#aktivasyon fonksiyonları
#karşılaştırma için 3 tanesini kullanacağız

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)


def initialize_weights_and_bias(inputs,layer_size,outputs):
    #weight ve biasları oluştur
    weight1 = np.random.randn(layer_size,inputs.shape[0]) * 0.1
    bias1 = np.zeros((layer_size,1))
    weight2 = np.random.randn(outputs.shape[0],layer_size) * 0.1
    bias2 = np.zeros((outputs.shape[0],1))
    wb_parameters = {"W1": weight1,
             "B1": bias1,
             "W2": weight2,
             "B2": bias2}
    return wb_parameters

def forward_propagation(x_train,wb_parameters,activation_function):
    # H, hiddenlayer
    H = np.dot(wb_parameters["W1"],x_train) + wb_parameters["B1"]
    #H nin outputu için activation function uygula
    if activation_function == "sigmoid":
        activateH = sigmoid(H)
    elif activation_function == "tanh":
        activateH = tanh(H)
    else :
        activateH = relu(H)
    # O, output
    O = np.dot(wb_parameters["W2"],activateH) + wb_parameters["B2"]
    # outpu sonucu almak için her zaman sigmoid activation function kullanacagız
    activateO = sigmoid(O)
    #geri yayılım ve hata hesabı için sonucları tutmamız lazım
    cache = {"H": H,
             "AH": activateH,
             "O": O,
             "AO": activateO}
    return cache


def compute_cost(output, y_train):
    #costu bul, errorda bulunabilirdi.
    logprobs = np.multiply(np.log(output), y_train) + np.multiply((1 - y_train), np.log(1 - output))
    cost = -np.sum(logprobs)/y_train.shape[1]
    
    acc = (100 - np.mean(np.abs(output - y_train)) * 100)
    return cost,acc

# Backward Propagation
def backward_propagation(wb_paramters,cache,x_train,y_train,activation_function):
    #deltayı bul gerekli türevleri bul ve çarp
    #gradientleri elde et
    delta = cache["AO"]- y_train
    gradient2 = np.dot(delta,cache["AH"].T)/x_train.shape[1]
    #bias güncelleyebilmek için delta sonularını topla
    dBias2 = np.sum(delta,axis =1,keepdims=True)/x_train.shape[1]
    derivativeH = np.dot(wb_paramters["W2"].T,delta)
    if activation_function == "sigmoid":
        derivativeH = derivativeH*(cache["AH"]*(1-cache["AH"]))
    elif activation_function == "tanh":
        derivativeH = derivativeH*(1 - np.power(cache["AH"], 2))
    else :
        derivativeH = relu(H)
    gradient1 = np.dot(derivativeH,x_train.T)/x_train.shape[1]
    dBias1 = np.sum(derivativeH,axis =1,keepdims=True)/x_train.shape[1]
    
    gradients = {"GW1": gradient1,
             "GB1": dBias1,
             "GW2": gradient2,
             "GB2": dBias2}
    return gradients

def update(parameters, grads, learning_rate):
    #parametreleri güncelle
    parameters = {"W1": parameters["W1"]-learning_rate*grads["GW1"],
                  "B1": parameters["B1"]-learning_rate*grads["GB1"],
                  "W2": parameters["W2"]-learning_rate*grads["GW2"],
                  "B2": parameters["B2"]-learning_rate*grads["GB2"]}
    
    return parameters


#prediction
def predict(model,x_test):
    # x_test sonuc bulmak için forward propagation yapılacak
    cache = forward_propagation(x_test,model["parameters"],model["act_func"])
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # sonuc 0.5'dan buyuk ise 1, kucuk ise 0 sonucunu verecek,
    for i in range(cache["AO"].shape[1]):
        if cache["AO"][0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    return Y_prediction,cache["AO"]

def acc_and_conf_matrix(prediction,expected,x_test):
    # accuracy için tahmin edilen ve olması gereken sonucların ortalamalarını bul
    print("Accuracy: {} %".format(100 - np.mean(np.abs(prediction - expected)) * 100))
    
    # confusion matrix çiz
    x = np.concatenate(expected,axis=0)
    y = np.concatenate(prediction,axis=0)
    y = y.astype(int)
    dictionary = {"x" : x,"y":y}
    df = pd.DataFrame(dictionary)
    confusion_matrix = pd.crosstab(df['x'], df['y'], rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True,fmt='g',cmap='Blues')

def roc(y_test,y_score):
    y_score = y_score.T
    y_test = y_test.T
  
    fpr , tpr , thresholds = roc_curve (y_test , y_score)
    plt.plot(fpr, tpr, label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    
def train(x_train, y_train,x_test,y_test,num_iterations=2500,hidden_layer_size=3,activation_func="tanh",learning_rate=0.01):
    cost_list = []
    trainacc_list = []
    testacc_list = []
    index_list = []
    model = {}
    #weight ve biasları oluştur
    parameters = initialize_weights_and_bias(x_train,hidden_layer_size, y_train)
    #epoch kadar weightleri güncelle ve öğret
    for i in range(0, num_iterations):
         # forward propagation
        cache = forward_propagation(x_train,parameters,activation_func)
        # costu hesapla
        cost,acc = compute_cost(cache["AO"], y_train)
         # backward propagation
        grads = backward_propagation(parameters, cache, x_train, y_train,activation_func)
         # parametreleri güncelle
        parameters = update(parameters, grads,learning_rate)
        #her 100 adımda costu yazdır
        
        if i % 20 == 0:
            model = {"parameters": parameters,"act_func": activation_func}
            predictions = predict(model,x_test)
            testacc_list.append((100 - np.mean(np.abs(predictions - y_test)) * 100))
            cost_list.append(cost)
            trainacc_list.append(acc)
            index_list.append(i)
            print ("Cost after iteration %i: %f , acc : %f" %(i, cost,acc))
    #cost fonskiyonunu çizdir
    plt.plot(index_list,cost_list,label='Cost')
    plt.xticks(index_list,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()
    
    #test and train fonksiyonunu cizdir
    plt.plot(index_list,testacc_list,label='Test',color ='r')
    plt.plot(index_list,trainacc_list,label='Train',color='g')
    plt.xticks(index_list,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
    return model

# train için verilebilecek parametreler
# num_iteration, verilmezse 2500 alınır
# hidden_layer_size, verilmezse 3 alınır
# activation_func, sigmoid, tanh ve relu verilebilir, verilmezse tanh alınır.
# learning_rate, verilmezse 0.01 alınır

# birinci model
# iterasyon = 500, learnig_rate = 0.1, hidden layer = 3
model = train(x_train, y_train,x_test,y_test,num_iterations=600,hidden_layer_size=5,learning_rate=0.025)
#model3 = train(x_train, y_train,num_iterations=1500,hidden_layer_size=5,activation_func="tanh",learning_rate=0.10)
#model4 = train(x_train, y_train,num_iterations=1500,hidden_layer_size=5,activation_func="sigmoid",learning_rate=0.10)
y_pred,y_score = predict(model,x_test)
roc(y_test,y_score)
acc_and_conf_matrix(y_pred,y_test,x_test)
