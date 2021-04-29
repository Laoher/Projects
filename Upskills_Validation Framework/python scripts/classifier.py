# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:10:49 2018

@author: BadrBelkeziz
badr.belkeziz@polytechnique.org
"""

import text_processing
import test_functional
from text_processing import process_content
from test_functional import process_file, generate_training_inputs, generate_validation_inputs, get_labels, generate_text_inputs
import frequencies
import keras
from test_functional import check_file

from keras import optimizers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import nltk
import gc
import h5py
print("version 3.0")
nltk.download()
def clear_all():
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]
    gc.collect()
if __name__ == "__main__":
    clear_all()   

def input_data(file_name,validation_file_name,mapping_table='mapping_table.csv',request_words="request_words.csv", include_other_words=True) :
    """Uses the functions from test_functional.py to create the  matrices for the neural network, for training and validation, input and output"""
    process_file(file_name,validation_file_name,mapping_table,request_words,new_data=True, include_other_words = True, occurences = True)
    print()
    print("We are transforming the data into matrices from the csv files, please wait it might take a few minutes ... ")
    gc.collect()
    start_time = time.time()
    generate_validation_inputs(validation_file_name),generate_training_inputs(file_name),get_labels(file_name,type_data = "train"), get_labels(validation_file_name, type_data = "validation")


def create_model (train_file_name,validation_file_name,mapping_table='mapping_table.csv', request_words = "request_words.csv",units = [16,16], regularization = False, regularizer = "l2", lambda_reg = 0.001, epochs = 1000, normalization = False, earlyStopping = False, thresholds = None, study_size = False, dropout = False , dropout_parameter = 0.5, patience = 20, include_other_words = True, multiple_categories = True ) :
    """ 1- I use global variables wasting time by getting the same data every time I  want to change my model 
     2- I try to avoid RAM issues by using H5 files instead of storing the iput matrices as a global variable 
     3- I create the neural network architecture, run the model, and get results, considering the paramters given by the user. You'll find here the documentations for the libraries I used : 
        Keras : https://keras.io/
        Tensorflow : https://www.tensorflow.org/ """
    print("Checking the training file ... " )
    flag_train = check_file(train_file_name, type_data = "train") 
    if (flag_train == 0 ) :
        print("No problem in training file")
    print("Checking the validation file ... " )
    flag_validation = check_file(validation_file_name, type_data = "validation")
    if(flag_validation == 0) :
        print("No problem in validation file")
    if (flag_train == 1 or flag_validation == 1) :
        print("Please check files and try again")
        return()
    
        
    HL = len(units)
    #1-
    global train_label
    global validation_label
    global sess
    gc.collect()
    if ('train_label' not in globals() ) : 
        input_data(train_file_name,validation_file_name,mapping_table,request_words,include_other_words)
        (train_label, validation_label) =(get_labels(train_file_name,type_data = "train"), get_labels(validation_file_name, type_data = "validation"))
        train_label = train_label.T
        validation_label = validation_label.T
    #2
    h5f_train = h5py.File("files/train/training_input.h5")
    train_data = h5f_train['train'][:]
    h5f_train.close()
    h5f_validation = h5py.File("files/validation/validation_input.h5")
    validation_data = h5f_validation['validation'][:]
    h5f_validation.close()
    shape = train_data.shape
    train_data = train_data.reshape(shape[0],1,shape[-1])
    shape = validation_data.shape
    validation_data = validation_data.reshape(shape[0],1,shape[-1])    

        
    print()
    print("-------------------------")
    print("Creating model ... Please wait, it might take a few minutes")
    start_time = time.time()
    
    #3
    
    if earlyStopping == True :
        callfunction = [keras.callbacks.EarlyStopping(monitor='val_loss', patience = patience)]
    else  : 
        callfunction = []

    if normalization == True :
      mean = train_data.mean(axis=0)
      std = train_data.std(axis=0)
      train_data = (train_data - mean) / std
      validation_data = (validation_data - mean) / std  
    model = keras.Sequential([
    keras.layers.Flatten(input_shape =(1,train_data.shape[-1])), #Check what is Flatten in tf
    ])
    if regularization == True :
        for i in range(HL) :
            model.add(keras.layers.Dense(units[i],kernel_regularizer = keras.regularizers.l2(lambda_reg), activation=tf.nn.relu))
            if dropout==True :
                model.add(keras.layers.Dropout(dropout_parameter))
    else :
        for i in range(HL) :
            model.add(keras.layers.Dense(units[i], activation=tf.nn.relu))
    if multiple_categories == True :
        activation= tf.nn.sigmoid
    else :
        activation = tf.nn.softmax
    if regularization == False :
        model.add(keras.layers.Dense(train_label.shape[1], activation=activation))
    else :
        model.add(keras.layers.Dense(train_label.shape[1],kernel_regularizer = keras.regularizers.l2(lambda_reg), activation=activation))
    sess = tf.Session()
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    if multiple_categories == True : 
        loss = 'binary_crossentropy'
    else :
        loss = 'categorical_crossentropy'
    
    
    model.compile(optimizer=opt,
              loss=loss,
              metrics=['accuracy'])
    sess.close()
    #model.summary()
    history = model.fit(train_data,
                    train_label,
                    epochs=epochs,
                    batch_size=512,
                    validation_data=(validation_data, validation_label),
                    shuffle=True,
                    verbose=0,
                    callbacks = callfunction)

    history_dict = history.history
    #print(history_dict)
    #print(history)
    history_dict.keys()

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print("final number of epochs : %s " %str(len(acc)))
    print()
    print("-------------------------")
    print()
    print("Final training loss : %s " %str(loss[-1]))
    print("Final validation loss : %s " %str(val_loss[-1]))
    epochs_list = range(1, len(acc) + 1)
    # "bo" is for "blue dot"
    plt.plot(epochs_list, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs_list, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("files/train/"+train_file_name+"_loss.png")
    plt.show()
    print("-------------------------")
    print()
    print("Final training accuracy : %s " %str(acc[-1]))
    print("Final validation accuracy : %s " %str(val_acc[-1]))
    epochs_list = range(1, len(acc) + 1)
    plt.plot(epochs_list, acc, 'go', label='Training accuracy')
    plt.plot(epochs_list, val_acc, 'g', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig("files/train/"+train_file_name+"_accuracy.png")
    plt.show()
    """" study_sie """
    print()
    print("-------------------------")
    print()
    if (study_size == True) :
        print("Studying the size of the training set ... please wait ")
    
    if study_size == True :
        size = []
        partial_acc = []
        partial_loss = []
        partial_val_acc=[]
        partial_val_loss=[]
        for i in range(1,len(train_data)) :
            partial_train = train_data[:i]
            partial_label = train_label[:i]
            size+=[i]
            partial_history = model.fit(partial_train,
                    partial_label,
                    epochs=epochs,
                    batch_size=512,
                    validation_data=(validation_data, validation_label),
                    shuffle=True,
                    verbose=0,
                    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience = patience)])

            partial_acc +=[partial_history.history['acc'][-1]]
            partial_val_acc+=[partial_history.history['val_acc'][-1]]
            partial_loss+=[partial_history.history['loss'][-1]]
            partial_val_loss+=[partial_history.history['val_loss'][-1]]
        plt.plot(size, partial_loss, 'ro', label='Training loss')
        plt.plot(size, partial_val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Size of the training set')
        plt.ylabel('Loss')
        plt.legend()

        plt.savefig("files/train/"+train_file_name+"_size_loss.png")

        plt.show()

        epochs_list = range(1, len(acc) + 1)
        plt.plot(size, partial_acc, 'ro', label='Training accuracy')
        plt.plot(size, partial_val_acc, 'r', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Size of the training set')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig("files/train/"+train_file_name+"_size_accuracy.png")
        plt.show()
    
    print("Execution time : %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    """metrics on training data """
    global predicted_trained
    num_classes = train_label.shape[-1]
    predicted_trained = model.predict(train_data)
    r_predicted = np.round(predicted_trained)
    #(np.round(predicted_trained*100))
    df_predicted = pd.DataFrame(predicted_trained)
    df_predicted.to_csv("files/train/results_"+train_file_name)
    #print(predicted_trained)
    sess=tf.Session()
    accur, acc_op = tf.metrics.accuracy(train_label,np.round(predicted_trained))
    if thresholds == None :
        fn, fn_op = tf.metrics.false_negatives(train_label,r_predicted)
        fp, fp_op = tf.metrics.false_positives(train_label,r_predicted)
        tp, tp_op = tf.metrics.true_positives(train_label,r_predicted)
        tn, tn_op = tf.metrics.true_negatives(train_label,r_predicted)
        prec, prec_op = tf.metrics.precision(train_label,r_predicted)
        recall, recall_op = tf.metrics.recall(train_label,r_predicted)
    else :
        fn, fn_op = tf.metrics.false_negatives_at_thresholds(train_label,predicted_trained,[thresholds])
        fp, fp_op = tf.metrics.false_positives_at_thresholds(train_label,predicted_trained,[thresholds])
        tp, tp_op = tf.metrics.true_positives_at_thresholds(train_label,predicted_trained,[thresholds])
        tn, tn_op = tf.metrics.true_negatives_at_thresholds(train_label,predicted_trained,[thresholds])
        prec, prec_op = tf.metrics.precision_at_thresholds(train_label,predicted_trained,[thresholds])
        recall, recall_op = tf.metrics.recall_at_thresholds(train_label,predicted_trained,[thresholds])
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.global_variables_initializer())
    print("------- Metrics on training set ---------")
    FN = (sess.run([fn,fn_op])[-1])
    FP = (sess.run([fp,fp_op])[-1])
    TP = (sess.run([tp,tp_op])[-1])
    TN = (sess.run([tn,tn_op])[-1])
    
    precision = (sess.run([prec,prec_op])[-1])
    recall = (sess.run([recall,recall_op])[-1])
    f1 = 2*precision*recall/(precision + recall)
    
    sess.close()
    print("True positives : %i " %int(TP))
    #print(TP)
    print("True negatives : %i " %int(TN))
    #print(TN)
    print("False negatives : %i " %int(FN))
    #print(FN)
    print("Falses positives : %i " %int(FP))
    #print(FP)
    print("Precision : %s " %str(precision))
    #print(precision)
    print("Recall : %s " %str(precision))
    #print(recall[0])  
    print("F1 Score : %s" %str(f1))
    #print(f1)
    print()
    print()
    print("If the model doesn't even fit well the training set, you're in a case of high bias. You may want to try a bigger network, or add additional features" )
    print()
    print()
    print("--------------")
            
    print("Execution time : %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    """metrics on validation data """
    global predicted
    num_classes = train_label.shape[-1]
    predicted = model.predict(validation_data)
    r_predicted = np.round(predicted)
    df_predicted = pd.DataFrame(predicted)
    df_predicted.to_csv("files/validation/results_"+validation_file_name)
 #   print(np.round(predicted*100))
    #print(predicted)
    sess=tf.Session()
    accur, acc_op = tf.metrics.accuracy(validation_label,np.round(predicted))
    if thresholds == None :
        fn, fn_op = tf.metrics.false_negatives(validation_label,r_predicted)
        fp, fp_op = tf.metrics.false_positives(validation_label,r_predicted)
        tp, tp_op = tf.metrics.true_positives(validation_label,r_predicted)
        tn, tn_op = tf.metrics.true_negatives(validation_label,r_predicted)
        prec, prec_op = tf.metrics.precision(validation_label,r_predicted)
        recall, recall_op = tf.metrics.recall(validation_label,r_predicted)
    else :
        fn, fn_op = tf.metrics.false_negatives_at_thresholds(validation_label,predicted,[thresholds])
        fp, fp_op = tf.metrics.false_positives_at_thresholds(validation_label,predicted,[thresholds])
        tp, tp_op = tf.metrics.true_positives_at_thresholds(validation_label,predicted,[thresholds])
        tn, tn_op = tf.metrics.true_negatives_at_thresholds(validation_label,predicted,[thresholds])
        prec, prec_op = tf.metrics.precision_at_thresholds(validation_label,predicted,[thresholds])
        recall, recall_op = tf.metrics.recall_at_thresholds(validation_label,predicted,[thresholds])
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.global_variables_initializer())
    print("------- Metrics on validation set ---------")
    FN = (sess.run([fn,fn_op])[-1])
    FP = (sess.run([fp,fp_op])[-1])
    TP = (sess.run([tp,tp_op])[-1])
    TN = (sess.run([tn,tn_op])[-1])
    
    precision = (sess.run([prec,prec_op])[-1])
    recall = (sess.run([recall,recall_op])[-1])
    f1 = 2*precision*recall/(precision + recall)
    
    sess.close()
    print("True positives : %i " %int(TP))
    #print(TP)
    print("True negatives : %i " %int(TN))
    #print(TN)
    print("False negatives : %i " %int(FN))
    #print(FN)
    print("Falses positives : %i " %int(FP))
    #print(FP)
    print("Precision : %s " %str(precision))
    #print(precision)
    print("Recall : %s " %str(precision))
    #print(recall[0])  
    print("F1 Score : %s" %str(f1))
    #print(f1)
    print("--------------")

    print("Execution time : %s seconds ---" % (time.time() - start_time))
    improvement(model,acc,val_acc,loss,val_loss,train_file_name,validation_file_name,mapping_table,request_words,units, regularization, regularizer, lambda_reg, epochs, normalization, earlyStopping, thresholds, study_size, dropout, dropout_parameter)
    
    
    
def improvement(model,acc,val_acc,loss,val_loss,train_file_name,validation_file_name,mapping_table,request_words,units, regularization, regularizer, lambda_reg, epochs, normalization, earlyStopping,thresholds, study_size, dropout, dropout_parameter) :
    """ Sort of a simple interface to ask the user to change the parameters in order to improve the architecture of the neural network """ 
    level_accuracy = acc[-1]-val_acc[-1]
    print("Final accuracy : %s ; Final validation accuracy : %s " %(str(acc[-1]), str(val_acc[-1])))
    print()

    todo = ''
    print()
    
    while  todo !='i' :    
        todo = str(input("Type 'p' to see the parameters of the model, 'save' to save the model if you are satisfied, 'i' to try and improve de model : "))
        if todo == 'p' :
            print()
            print("Regularization : %s ; Regularization's parameter : %s " %(str(regularization),str(lambda_reg))) 
            print("Epochs : %i " %epochs)
            print("Standardization:  %s" %str(normalization))
            print("EarlyStopping : %s" % str(earlyStopping))
            print("Thresholds : %s " %str(thresholds))
            print("Dropout : %s , dropout_parameter : %s " %(str(dropout),str(dropout_parameter)))
        elif todo == 'save' :
            name = train_file_name
            model.save("files/models/"+name+'.h5')
            return()
    
        
    if (len(acc)==epochs) :
        print("---------------")
        print()
        print("The algorithm did %s iterations as you asked for. If you notice that the validation loss function is still decreasing, you may be in a case of underfitting." %str(epochs))
        print("Otherwise, if the validation loss started increasing or levelling, you may be in a case of overfitting.")
    if (earlyStopping == True ) :
        print("The earlyStopping is set to true. Try setting it to False to know if you are in a case of underfitting or overfitting. Type True if you know in which case you are")
        earlyStopping = (input("EarlyStopping : (True or False) ")=='True')
        #print(earlyStopping)
        #print(type(earlyStopping))
    case = str(input("Please let us know in which case you are ('o' for overfitting, 'u' for underfitting, 'r' if you don't know) : "))
    if case == 'r' :
        create_model (train_file_name,validation_file_name,mapping_tablerequest_words,units, regularization, regularizer, lambda_reg, epochs, normalization, earlyStopping,thresholds, study_size, dropout, dropout_parameter,patience, include_other_words, multiple_categories ) 
    if case =='o' :
        print("You are in a case of overfitting. You may want to : " )
        print(" - Standardize your data:  ")
        normalization = (input("Standardizing (True or False) : ")=='True')
        print()
        print(" - Set the earlyStopping to True :  ")
        earlyStopping = (input("EarlyStopping (True or False) : ")=='True')
        print()
        print(" - Reduce the number of hidden layers / units per layer ")
        print("Current architecture : ")
        print("   -  %i hidden layer(s) " %len(units))
        print("   -  units : %s " %(",".join([str(i) for i in units])))
        units = input("List of units per hidden layer (Example : 16,32,8 ) : ")
        units = [int(n, 10) for n in units.split(",")]
        print()
        print(" - Add/increase regularization" )
        print("Current regularization : ")
        print("Regularization : %s " %str(regularization))
        print("Lambda : %s " %str(lambda_reg))
        regularization = (input("l2 regularization (True or False ) : ")=='True')
        if regularization == True :
            lambda_reg = float(input("Lambda parameter for regularization : "))
        print()
        print(" - Add a dropout layer " )
        print("Dropout : %s " %str(dropout))
        print("Dropout rate : %s " %str(dropout_parameter))
        dropout = (input("Set dropout (True  or False ) : ")=='True')
        #print(dropout)
        if  dropout == True :
            dropout_parameter = float(input("Dropout rate : "))
        print()
        print()
        print("In case of high variance, you should try to get more training data")
        print()
        print()
        study_size=(input("Type 'y' if you want to study the size of the training set :  ")=='y')
        create_model (train_file_name,validation_file_name,mapping_table,request_words,units, regularization, regularizer, lambda_reg, epochs, normalization, earlyStopping,thresholds, study_size, dropout, dropout_parameter,patience, include_other_words, multiple_categories = multiple_categories
                      ) 
    elif case == 'u' :
        print("You are in a case of underfitting. You can either : " )
        print(" - Standardize your data:  ")
        normalization = (input("Standardizing (True or False) : ")=='True')
        print()
        print(" - Set the earlyStopping to False :  ")
        earlyStopping = (input("EarlyStopping (True or False) : ")=='True')
        print()
        print(" - Increase the number of hidden layers / units per layer ")
        print("Current architecture : ")
        print("   -  %i hidden layer(s) " %len(units))
        print("   -  units : %s " %(",".join([str(i) for i in units])))
        units = input("List of units per hidden layer (Example : 16,32,8 ) : ")
        units = [int(n, 10) for n in units.split(",")]
        print(" - Increase the number of epochs. The algorithm is currently running %i epochs " %epochs)
        epochs = int(input("Number of epochs : "))
        print(" We suggest you do not change the regularization" )
        print("Current regularization : ")
        print("Regularization : %s " %str(regularization))
        print("Lambda : %s " %str(lambda_reg))
        regularization = (input("l2 regularization (True or False ) : ")=='True')
        if regularization == True :
            lambda_reg = float(input("Lambda parameter for regularization : "))
        print()
        print(" We suggest you do not change the dropout layer " )
        print("Dropout : %s " %str(dropout))
        print("Dropout rate : %s " %str(dropout_parameter))
        dropout = (input("Set dropout (True  or False ) : ")=='True')
        #print(dropout)
        if  dropout == True :
            dropout_parameter = float(input("Dropout rate : "))
        print()
        print()
        print("In case of high bias, you should try to add additional features")
        print()
        print()
        study_size=(input("Type 'y' if you want to study the size of the training set :  ")=='y')
        create_model (train_file_name,validation_file_name,mapping_table,request_words,units, regularization, regularizer, lambda_reg, epochs, normalization, earlyStopping,thresholds, study_size, dropout, dropout_parameter,patience, include_other_words, multiple_categories) 


def use_model(model_name, test_file_name):
    """ Uses the model on the test set, store the results into a csv file"""
    model = keras.models.load_model("files/models/"+model_name+".h5")
    test_data = generate_text_inputs(test_file_name, model_name)
    shape = test_data.shape
    test_data = test_data.reshape(shape[0],1,shape[-1])
    model.summary()
    df = pd.DataFrame(np.round(model.predict(test_data)))
    df.to_csv("files/test/results_category_"+test_file_name)
    print(np.round(model.predict(test_data))    )
    
def analyze_model(model_name) :
    """ Allows the user to see the architecture of the neural network saved in the model """
    model = keras.models.load_model("files/models/"+model_name+".h5")
    for layer in model.get_config() :
        print("Layer : %s " % layer['class_name'])
        for config in layer['config'].keys() :
            print("       %s : %s " %(config,layer['config'][config]))
        
    print("-----")
    print(model.summary())
    print("------")


gc.collect()

regularizer = "l2"
lambda_reg = 0.001
normalization = False
earlyStopping = False
thresholds = None
study_size = False 
dropout = False 
dropout_parameter= 0.5
patience = 20
include_other_words = True
train_file_name = str(input("Please input the name of the training file (e.g : Requirements.csv) : "))
validation_file_name = str(input("Please input the name of the validation file (e.g : validation.csv) : "))
multiple_categories =(str(input("multiple categories ? (y/n) : "))=='y')
mapping_table = str(input("Please input the name of the mapping file (default : mapping_table.csv) : "))
if (mapping_table == '') :
    mapping_table = "mapping_table.csv" 
request_words = str(input("Please input the name of the request file (default : request_words.csv) : "))
if (request_words == '') :
    request_words = "request_words.csv"
epochs = input("Number of epochs for the model (defaults : 800 ) : ")
if (epochs == '') :
    epochs = 800
else :
    epochs = int(epochs)
units = input("List of units per hidden layer (defaults : 100,100 ) : ")
if (units == '') :
    units = [100,100]
else :
    units = [int(n, 10) for n in units.split(",")]
regularization = (input("l2 regularization (True or False ) : ")=='True')
if regularization == True :
            lambda_reg = float(input("Lambda parameter for regularization : "))
dropout = (input("Set dropout (True  or False ) : ")=='True')
if  dropout == True :
    dropout_parameter = float(input("Dropout rate : "))
create_model (train_file_name,validation_file_name,mapping_table,request_words, units , regularization, regularizer, lambda_reg , epochs, normalization , earlyStopping , thresholds, study_size , dropout , dropout_parameter, patience, include_other_words, multiple_categories)


