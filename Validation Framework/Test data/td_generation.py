# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:50:20 2018

@author: BadrBelkeziz
"""

import pandas as pd
import re


def process(filename, important_cat) :
    df=pd.read_csv(filename + '.csv', header =0, index_col = False);
    df_dict = df.to_dict()
    transposed_df_dict = df.T.to_dict()
    
    #print(df_dict)
    charac_important ={}
    for i in important_cat :
        charac_important[i]=df_dict[str(i)]
    i=0
    print(charac_important)
    charac_to_consider = {}
    for charac in charac_important.values () :
        #print(charac)
        charac_to_consider[i]={}
        j=0
        for ch in charac.values() :
            if ch not in charac_to_consider[i].values() :
                charac_to_consider[i][j] = ch
                j+=1
        i+=1
    print (charac_to_consider)
    elt_dict = {}
    k=0
    for i in range(len(important_cat)) :
        c = charac_to_consider[i]
        list_charac = df_dict[str(important_cat[i])]
        print(list_charac)
        for char_ in c.values() :
            for  elt in list_charac.keys():
                if(char_ == list_charac[elt]) :
                    if(df_dict[str(0)][elt] not in elt_dict.values()) :
                        elt_dict[k]=df_dict[str(0)][elt]
                        k+=1
                    break;
    return(elt_dict)
                    
            
def group(elt_dict1,elt_dict2) : #requires len(elt_dict1)==len(elt_dict2)
    group_dict = {}
    if (len(elt_dict1)==0) :
        return(elt_dict2)
    elif (len(elt_dict2)==0) :
        return(elt_dict1)
    for i in range(len(elt_dict1)) :
        group_dict[i] = elt_dict1[i]+";"+elt_dict2[i]
    return(group_dict)

def combine(elt_dict1,elt_dict2) :
    combine_dict = {}
    if (len(elt_dict1)==0) :
        return(elt_dict2)
    elif (len(elt_dict2)==0) :
        return(elt_dict1)
    k=0
    for i in range(len(elt_dict1)) :
        for j in range(len(elt_dict2)) :
            combine_dict[k] = elt_dict1[i]+";"+elt_dict2[j]
            k+=1
    return(combine_dict)
    
    
def process_permut(filename, important_cat) :
    df=pd.read_csv(filename + '.csv', header =0, index_col = False);
    columns = df.columns.values.tolist()
    df_dict = df.to_dict()
    #print(df_dict)
    list_charac = {}

    for i in important_cat :
        list_charac = group(list_charac,df_dict[str(i)])
    #print(list_charac)
    
  
    charac_important ={}
    for i in important_cat :
        charac_important[i]=df_dict[str(i)]
    i=0
    charac_to_consider = {}
    for charac in charac_important.values () :
        #print(charac)
        charac_to_consider[i]={}
        j=0
        for ch in charac.values() :
            if ch not in charac_to_consider[i].values() :
                charac_to_consider[i][j] = ch
                j+=1
        i+=1
    #print (charac_to_consider)
    
    important_charac ={}
    for i in charac_to_consider.keys() :
        important_charac = combine(important_charac,charac_to_consider[i])
    #print(important_charac)
        
        
    elt_dict = {}
    k=0
    for i in important_charac.keys() :
        c = important_charac[i]
        flag = 0 
        for  elt in list_charac.keys():
            if(c == list_charac[elt]) :
                elt_dict[k]=df_dict[str(0)][elt]
                k+=1
                flag = 1
                break;
        if (flag == 0) :
            print("%s not found" % c)
    return(elt_dict)
    
def combine_files(elt_dict1,elt_dict2) :
    combine_dict = {}
    if (len(elt_dict1)==0) :
        return(elt_dict2)
    elif (len(elt_dict2)==0) :
        return(elt_dict1)
    k=0
    for i in range(len(elt_dict1)) :
        for j in range(len(elt_dict2)) :
            if type(elt_dict1[i]) == str :
                combine_dict[k] = {0 : elt_dict1[i], 1 : elt_dict2[j] }
                k+=1
            elif (type(elt_dict1[i])==dict) :
                elt_to_combine = elt_dict1[i].copy() # Make a copy
                elt_to_combine[len(elt_dict1[i])] = elt_dict2[j]
                combine_dict[k] = elt_to_combine
                k+=1
    return(combine_dict)
    
def generate_td() :
    df=pd.read_csv('static_data.csv', header =0, index_col = False);
    static_data_dict = {}
    for index,row in df.iterrows() :
        static_data_dict[index] = {0 : row["Static Data"], 1 : [int(i) for i in row["Characteristic"].split(";")], 2 : row["Permutations"]}
    sd_generation = {}    
    for sd in static_data_dict.values():
        if sd[2] ==1 :
            processing = process_permut
        elif sd[2] == 0 :
            processing = process
        else :
            print("Invalid value permutation 1 or 0 : " + sd[0])
        #print(sd)
        sd_list = processing(sd[0],sd[1])
        #print(sd_list)
        sd_generation = combine_files(sd_generation,sd_list)
        #print(sd_generation)
    return(sd_generation)
   
    
def new_process(filename, important_cat) :
    df=pd.read_csv(filename + '.csv', header =0, index_col = False);
    columns = df.columns.values.tolist()
    df_dict = df.to_dict()
    #print(df_dict)
    transposed_df_dict = df.T.to_dict()
    #print(transposed_df_dict)
    #print(df_dict)
    charac_important ={}
    for i in important_cat :
        charac_important[i]=df_dict[str(i)]
    i=0
    #print(charac_important)
    charac_to_consider = {}
    for charac in charac_important.values () :
        #print(charac)
        charac_to_consider[i]={}
        j=0
        for ch in charac.values() :
            if ch not in charac_to_consider[i].values() :
                charac_to_consider[i][j] = ch
                j+=1
        i+=1
    #print (charac_to_consider)
    elt_dict = {}
    k=0
    for i in range(len(important_cat)) :
        c = charac_to_consider[i]
        list_charac = df_dict[str(important_cat[i])]
        #print(list_charac)
        for char_ in c.values() :
            for  elt in list_charac.keys():
                if(char_ == list_charac[elt]) :
                    if(df_dict[str(0)][elt] not in elt_dict.values()) :
                        
                        static_data = transposed_df_dict[elt]
                        for j in range(len(important_cat)-1) :
                            data_to_erase = static_data[str(j+1)]
                            elt_dict_copy = elt_dict.copy()
                            for sd in elt_dict_copy.values() :
                                key = find_key_in_dict(df_dict['0'],sd)
                                if df_dict[str(j+1)][key]==data_to_erase:
                                    del elt_dict[find_key_in_dict(elt_dict,sd)]        
                        elt_dict[k]=df_dict[str(0)][elt]
                        k+=1
                    break;
    return(elt_dict)
    
def find_key_in_dict(dict_,elt) :
    for key in dict_.keys() :
        if (dict_[key] == elt) :
            return (key)
    return (" Element not found in dict")
    
                    