# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 10:15:03 2018

@author: BadrBelkeziz
badr.belkeziz@polytechnique.org
"""

import pandas as pd
from nltk.probability import FreqDist
import frequencies
from frequencies import jsdict
from text_processing import process_content
import time
import json
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from numpy import array
import re
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import math
import text_processing
from nltk.corpus import stopwords
from nltk.tokenize import MWETokenizer
import h5py


ps = PorterStemmer()
stop_words = set(stopwords.words('english')+['The','``',',','.',';',':','&','"','-','>','<','*','(',')','the','-','–','+','?', '‘', '!', '..', '--', '_', '|', '[', ']', '#', '//',"'\'" ,"/", '$', '^', '{', '}', '~','“','«', '»','s'])


                        

        
def check_file(file_name,type_data= "train") :
    if type_data == "train" :
        file_path = "files/train/"+file_name
    if type_data == "validation" :
        file_path = "files/validation/"+file_name
    df = pd.read_csv(file_path)
    columns_file = df.columns.values.tolist()
    df_dict = df.to_dict()
    data_dict = {}
    texts = df_dict[columns_file[2]]
    theme = df_dict[columns_file[1]]
    flag = 0
    for i in range(len(texts)) :
        flag_text = 0 
        if type(texts[i])==str :
            for j in range(3,len(columns_file)) :
                if (flag_text == 0 and df_dict[columns_file[j]][i] !=0 and  df_dict[columns_file[j]][i] != 1) :
                    print("Problem in row %s : category %s not equal to 0 or 1. " %(str(i), columns_file[j]))
                    flag_text = 1
                    flag = 1
                    
        else :
            for j in range(3,len(columns_file)) :
                if(flag_text == 0 and not math.isnan(df_dict[columns_file[j]][i])) :
                    print("Problem in row %s : category given but no text" %(str(i)))
                    flag = 1
                    flag_text = 1 
                    
    return(flag)
            
    
                 
                 
def process_file(file_name, validation_file_name, mapping_table='mapping_table.csv', request_words="request_words.csv",new_data=True, include_other_words = True, occurences = True) :
    """ Data transformation :
        This function aims at creating a dictionary of frequencies from a text using various basic NLP technique.
        - Mapping : transform a word into its tarfet, listed in the file mapping_table.csv
        - Look for requested words/group of words, reject unwanted words
        - Using tokenizing, stemming, lemmatizing. In this version I do not use chunking but you can see that the function process_content() in the file text_processing.py can do it """ 

    start_time = time.time()
    """ Prepare mapping """
    global map_dict
    data_mapping = pd.read_csv("files/train/"+mapping_table);
    map_dict = data_mapping.to_dict()
    words = map_dict['word']
    
    
    """ Prepare dict of frequencies """
    if new_data==True :
        json_str = "{}"             
    else : 
        with open("files/train/json_functional_frequencies_"+file_name+".txt", "r", encoding = "utf-8-sig") as file:
            json_str = file.read()
    dic_freq = json.loads(json_str)
    try :
        dataf=pd.read_csv('files/train/frequencies_'+file_name);
        x=dataf['file name'].values.tolist()
    except Exception as e :
        x=[]
    
    """ Prepare requested words """
    global requested
    global rejected
    data_request = pd.read_csv("files/train/"+request_words);
    request_dict = data_request.to_dict()
    columns_req = data_request.columns.values.tolist()
    requested = request_dict[columns_req[0]]
    rejected = request_dict[columns_req[1]]
    rejected = [i for i in rejected.values() if type(i) == str ]
    requested = [i for i in requested.values() if type(i) == str ]


        
    requested_groups = [tuple([lemmatizer.lemmatize(i.lower()) for i in word_tokenize(k)]) for k in requested]
    requested =  [" ".join(list(k)) for k in requested_groups]  
    rejected_groups = [tuple([lemmatizer.lemmatize(i.lower()) for i in word_tokenize(k)]) for k in rejected]
    rejected =  [" ".join(list(k)) for k in rejected_groups]   
    tokenizer = MWETokenizer(requested_groups+rejected_groups, separator = " ")
    for i in requested:
        dic_freq[i] = []
    
    """ Process file """ 
    df=pd.read_csv("files/train/"+file_name);
    columns_file = df.columns.values.tolist()
    df_dict = df.to_dict()
    data_dict = {}
    texts = df_dict[columns_file[2]]
    theme = df_dict[columns_file[1]]
    


    all_words = {}    
    for i in range(len(theme)):
        try :
            text = texts[i]
            word_tokens = word_tokenize(text)
            word_tokens = tokenizer.tokenize([lemmatizer.lemmatize(w.lower()) for w in word_tokens])       

            if include_other_words == True :
                filtered_sentence = [w.lower() for w in word_tokens if not w in stop_words and not re.match("([0-9]+)",w) and not re.match("/",w) and not re.match("©",w) and not w in rejected ]
            else :
                filtered_sentence = [w.lower() for w in word_tokens if not w in stop_words and not re.match("([0-9]+)",w) and not re.match("/",w) and not re.match("©",w) and not w in rejected and w in requested ]
           
            words_map = filtered_sentence
            for j in range(len(words_map)) :
                words_map[j] = process_mapping(words_map[j],words)
                word = words_map[j]
                words_map[j] = ps.stem(words_map[j])

                        
                if (words_map[j] not in all_words.values()) :
                    try :  
                        index = list(all_words.keys())[-1]
                        index+=1
                        all_words[index]= words_map[j]
                    except Exception as e : 
                        index=0
                        all_words[index]= words_map[j]
            texts[i] = (' ').join(lemmatizer.lemmatize(w) for w in words_map)
            #print(texts[i])
            #print(texts[i])
            #print(process_content(texts[i]))
            if ("file name" in dic_freq.keys()) :
                dic_freq["file name"] += ["text " + str(theme[i])]
                nbr_txt = len(dic_freq["file name"])
            else :
                dic_freq["file name"]= ["text " + str(theme[i])]
                nbr_txt = 0
                nbr_txt = len(dic_freq["file name"])
            try : 
                freq = FreqDist(words_map).most_common(len(texts[i]))

                if (occurences == False) :
                    for w in freq :
                        if w[1] > 1 :
                            w[1] = 1
                for w in freq :
                    if (w[0] in dic_freq.keys()) :
                        dic_freq[w[0]] += [w[1]]
            
                    else :
                        dic_freq[w[0]] = [0]*(nbr_txt-1)        
                        dic_freq[w[0]]+=[w[1]]
                    
                for w in dic_freq.keys():
                    l=len(dic_freq[w])
                    if (l!=nbr_txt) :
                        dic_freq[w]+=[0]*(nbr_txt-l)
            #print(fdist) 
            except TypeError :
                print("type error")
            print("row %i added " % i)
        except TypeError as e:
            print("No text in row %i" %i )
    print(" %s texts added to the training set " % str(len(dic_freq["file name"])))
    print(" %s features added to the training set " % str(len(dic_freq)-1))
    print("Execution time : %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    print("Storing the frequencies into a csv file ... ")
    print()
    del dic_freq["file name"]
    with open("files/train/json_functional_frequencies_"+file_name+".txt", "w", encoding = "utf_8") as file:
        file.write(str(jsdict(dic_freq)))
    df = pd.DataFrame(dic_freq, columns = [k for k in dic_freq.keys()])
    df.to_hdf("files/train/"+file_name+".hdf5", key = "train", mode = 'w')
    df.to_csv("files/train/frequencies_"+file_name) 
    print("Execution time : %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    validation_data(list(dic_freq.keys()),validation_file_name,words)
        
   

def validation_data (list_train_words,validation_file_name, mapping, validation = True,request_words="request_words.csv", include_other_words = True):
    """ Same function as process_file but adapted to creatinga matrix of validation data.
    The difference is that when we process the training set, we can take into account any word we want, with the validation set, we can only consider the words already in the training set.
    For the Neural network the be logic, both matrices of validation and training must mean the same thing.
    This function can also be used to generate a matrix for a test set, since it is the same principle"""
    global requested
    global rejected

    data_request = pd.read_csv("files/train/"+request_words);
    request_dict = data_request.to_dict()
    columns_req = data_request.columns.values.tolist()
    requested = request_dict[columns_req[0]]
    rejected = request_dict[columns_req[1]]
    rejected = [i for i in rejected.values() if type(i) == str ]
    requested = [i for i in requested.values() if type(i) == str ]



        
    requested_groups = [tuple([lemmatizer.lemmatize(i.lower()) for i in word_tokenize(k)]) for k in requested]
    requested =  [" ".join(list(k)) for k in requested_groups]  
    rejected_groups = [tuple([lemmatizer.lemmatize(i.lower()) for i in word_tokenize(k)]) for k in rejected]
    rejected =  [" ".join(list(k)) for k in rejected_groups]   
    tokenizer = MWETokenizer(requested_groups, separator = " ")
    global validation_freq
    validation_freq = {"Num" : []}
    for i in list_train_words :
        validation_freq[i]=[]
    try :
        del validation_freq["file name"]
    except Exception as e :
        print()
    #print(validation_freq)
    if validation == True :
        validation_df=pd.read_csv("files/validation/"+validation_file_name);
    else :
        validation_df=pd.read_csv("files/test/"+validation_file_name);
    columns = validation_df.columns.values.tolist()
    validation_df_dict = validation_df.to_dict()
    data_dict = {}
    texts = validation_df_dict[columns[2]]
    theme = validation_df_dict[columns[1]]
    #print(theme)

    start_time = time.time()

    all_words = dict({})    
    #print(validation_freq)
    for i in range(len(theme)):
        try :
            text = texts[i]
            word_tokens = word_tokenize(text)
            word_tokens = tokenizer.tokenize([lemmatizer.lemmatize(w.lower()) for w in word_tokens])       
            #print(word_tokens)
            #print(stop_words)
            #words_stem = [ps.stem(w) for w in words_map]
            if include_other_words == True :
                filtered_sentence = [w.lower() for w in word_tokens if not w in stop_words and not re.match("([0-9]+)",w) and not re.match("/",w) and not re.match("©",w) and not w in rejected ]
            else :
                filtered_sentence = [w.lower() for w in word_tokens if not w in stop_words and not re.match("([0-9]+)",w) and not re.match("/",w) and not re.match("©",w) and not w in rejected and w in requested]
            #print(filtered_sentence)
            #print(words_map)           
            words_map = filtered_sentence
            words_ref_in_list = [""]*len(words_map)
            for j in range(len(words_map)) :
                flag = 0
                words_map[j] = process_mapping(words_map[j],mapping)
                words_map[j] = ps.stem(words_map[j])
                word = words_map[j]
                """ finaly I chose not to use the following lines, not efficient enough in order to find synonyms in the words we read already """
                #for l in range(len(list_train_words)) : 
                #    word_ref = list_train_words[l]
                #    for synset in wn.synsets(word_ref) :
                #        for lemma in synset.lemmas() :
                #            if (lemma.name()==word or ps.stem(lemma.name())==ps.stem(word)) :
                #                #print (lemma.name() + " === " + word_ref)
                #                words_map[j]=word_ref
                #                words_ref_in_list[j]=list_train_words[l]
                #                flag = 1
            #                  words_map[i] = ps.stem(words_map[i])
                #    if (ps.stem(word_ref)==ps.stem(word)) :
                #        words_map[j]=word_ref
                #        words_ref_in_list[j] = list_train_words[l]
                #        flag=1
                #if flag == 0 :
                #    words_ref_in_list[j]=word
                #print(all_words)
                #if (words_map[j] not in all_words.values()) :
                #    try :  
                #        index = list(all_words.keys())[-1]
                #        index+=1
                #        all_words[index]= words_map[j]
                #    except Exception as e : 
                #        index=0
                #        all_words[index]= words_map[j]
            texts[i] = (' ').join(lemmatizer.lemmatize(w.lower()) for w in words_map)
 #           print(words_ref_in_list)
            freq = FreqDist(words_map).most_common(len(texts[i]))
 #           print(freq)
            #print(freq)
            #print(words_ref_in_list)
            #print(len(freq)==len(words_ref_in_list))
            #texts[i] = (' ').join(lemmatizer.lemmatize(w) for w in words_map)
            #freq = FreqDist(process_content(texts[i])).most_common(len(texts[i]))
            #print(freq)
            for key in range(len(freq)) :
                w=freq[key]
                if (w[0] in validation_freq.keys()) :
                    validation_freq[w[0]] += [w[1]]
                else :
                    print("%s not in training set " %(w[0]))
            validation_freq["Num"] +=[i]
            nbr_txt = len(validation_freq["Num"])
            for key in validation_freq.keys() :
                if len(validation_freq[key])!=nbr_txt:
                        validation_freq[key] += [0]
            print("row %i added " % i)
        except Exception as e :
            print("No text in row %i" %i )
    del validation_freq["Num"]
    print("Execution time : %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    print("Storing the frequencies into a csv file ... ")
    print()
    df = pd.DataFrame(validation_freq, columns = [k for k in validation_freq.keys()])
    df.to_hdf("files/validation/"+validation_file_name+".hdf5", key = "validation", mode = 'w')
    if validation == True : 
        df.to_csv("files/validation/frequencies_"+validation_file_name)   
    else : 
        df.to_csv("files/test/frequencies_" + validation_file_name)  
    print("Execution time : %s seconds ---" % (time.time() - start_time))

            
def get_labels (file_name, type_data = "train"):
    """ Process the training of the validation file by getting only the labels (columns[>=3]) in the file """
    start_time = time.time()
    """ Process file """ 
    if (type_data == "train"):
        df=pd.read_csv("files/train/"+file_name, header=None)
    else :
        df=pd.read_csv("files/validation/"+file_name, header = None);
    columns = df.columns.values.tolist()
    df_dict = df.to_dict() 
    del df_dict[0]
    del df_dict[1]
    del df_dict[2]
    df_copy ={}
    for i in range(3,len(df_dict)+3) :
        del df_dict[i][0]
        df_copy[i] = df_dict[i].copy()
    for i in range(3,len(df_dict)+3) :
        for j in df_dict[i].keys():
            try : 
                df_copy[i][j] = int(df_copy[i][j])
            except ValueError :      
                del df_copy[i][j]
    
    for i in df_copy.keys() :
        df_copy[i] = np.array(list(df_copy[i].values()))
    labels = np.array(list(df_copy.values()))

    label_matrix = np.zeros((labels.shape[0],labels[0].shape[0]))
    for i in range(labels.shape[0]) :
        label_matrix[i] = labels[i]
    print("%s labels done" %(type_data))
    print("Execution time : %s seconds ---" % (time.time() - start_time) )
    return(label_matrix)
    


def process_mapping (word,words) :
    """ Transform the word into its target, see the file mapping word_csv """
    if (word not in words.values()) :
        return (word)
    else : 
        target = map_dict['target']
        for key in words.keys() :
            if words[key] == word:
                return (target[key])
            
def generate_training_inputs(file_name) : 
    """ Reads the DataFrame  stored into an HDF5 file by the function process_file(), create a numpy matrix from it and store it in an h5 file """
    start_time = time.time()
    #data = training_dataFrame
    data = pd.read_hdf("files/train/"+file_name+".hdf5", key = "train")
    #data= pd.read_csv('files/train/frequencies_'+file_name, header=0, index_col=False)
    #data.corr().to_csv("files/train/correlation_"+file_name)
    nbr_columns = len(list(data.columns))    
    matrix = array(data.values.T[1:nbr_columns],dtype=np.float64).T
    h5f = h5py.File('files/train/training_input.h5','w')
    h5f.create_dataset('train', data = matrix)
    print("Training data done : execution time : %s seconds ---" % (time.time() - start_time) )
    
def generate_validation_inputs(validation_file_name) : 
    """ Reads the DataFrame  stored into an HDF5 file by the function validation_data(), create a numpy matrix from it and store it in an h5 file """
    start_time = time.time()
    #data = validation_dataFrame
    data = pd.read_hdf("files/validation/"+validation_file_name+".hdf5", key = "validation")
    #data= pd.read_csv('files/validation/frequencies_'+validation_file_name, header=0, index_col=False)
    nbr_columns = len(list(data.columns))
    #data.corr().to_csv("files/validation/correlation_"+validation_file_name)
    matrix = array(data.values.T[1:nbr_columns],dtype=np.float64).T
    h5f = h5py.File('files/validation/validation_input.h5','w')
    h5f.create_dataset('validation', data = matrix)
    print("Validation data done : execution time : %s seconds ---" % (time.time() - start_time) )
    return(matrix)
    
def generate_text_inputs(file_name, train_file_name, mapping='mapping_table.csv', request_words = "request_words.csv") :
    """ Prepare and launch the function validation_data() with the parameter validation = False """ 
    """ Prepare requested words """
    global rejected
    data_request = pd.read_csv("files/train/"+request_words);
    request_dict = data_request.to_dict()
    columns_req = data_request.columns.values.tolist()
    #print(request_dict)
    requested = request_dict[columns_req[0]]
    rejected = request_dict[columns_req[1]]
    #print(rejected.values())
    for i in requested.values() :
        if not math.isnan(i) :
            dic_freq[i] = []
    
    """ Prepare mapping """
    global map_dict
    data_mapping = pd.read_csv("files/train/"+mapping);
    map_dict = data_mapping.to_dict()
    words = map_dict['word']
    #print(words)
    train_data= pd.read_csv('files/train/frequencies_' + train_file_name, header=0, index_col=False)
    list_words=(list(train_data.to_dict().keys())[1:])
    validation_data(list_words, file_name, words, validation = False)
    data=pd.read_csv("files/test/frequencies_"+file_name, header=0, index_col=False)
    data.corr().to_csv("files/test/correlation_"+file_name)
    nbr_columns = len(list(data.columns))   
    matrix = array(data.values.T[1:nbr_columns],dtype=np.float64).T
    return(matrix)
    

    

    
    
        