"""" File reading (csv and txt) and processing"""
import numpy
import nltk
#import gensim
#import pandas as pnd
#import os
import csv
#import sys
#import pprint
import re
#from gensim import corpora, models, similarities
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import treebank_chunk
from nltk.tokenize import MWETokenizer
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
## Traitement du fichier design.txt

#stop_words = set(stopwords.words('english')+[',','.',';',':','&','"','-','>','<','*','(',')','the','All','-','The','–','+','?','AND','this','This','also','You' '‘', '!', '..', '--', '_', '|', '[', ']', '#', '\\', '$', '^', '{', '}', '~','“','«', '»',])
stop_words = set(stopwords.words('english')+['The','``',',','.',';',':','&','"','-','>','<','*','(',')','the','-','–','+','?', '‘', '!', '..', '--', '_', '|', '[', ']', '#', '//',"'\'" ,"/", '$', '^', '{', '}', '~','“','«', '»','s'])
ps = PorterStemmer()

def process_mapping (word) :
    """Transform word into its target, listed in mapping_table.csv"""
    
    data_mapping = pd.read_csv('mapping_table.csv');
    map_dict = data_mapping.to_dict()
    words = map_dict['word']
    #print(words)
    
    if (word not in words.values()) :
        return (word)
    else : 
        target = map_dict['target']
        for key in words.keys() :
            if words[key] == word:
                return (target[key])

def txt_processing(file_name) :
    "returns the frequencies of the words in the text. Not used in the final algorithm"
    file_name = str(file_name)+'.txt'
    try :
        file  = open("files/"+file_name,"r")
        DESIGN_TEXT = file.read()
    except UnicodeDecodeError : 
        file  = open("files/"+file_name,"r", encoding = "utf-8")
        DESIGN_TEXT = file.read() 
    word_tokens = word_tokenize(DESIGN_TEXT)

    filtered_sentence = [w for w in word_tokens if not w in stop_words and not re.match("([0-9]+)",w) and not re.match("/",w) and not re.match("©",w) ]
    words_stem = [ps.stem(w) for w in filtered_sentence]
    fdist_stem = FreqDist(words_stem)

    words_lemmatize= [lemmatizer.lemmatize(w) for w in filtered_sentence]
    
    fdist = FreqDist(words_stem)



    return (fdist.most_common(len(words_stem)))
 

def process_content(text) :
    """ Tokenizing text according to the chunk you give in 'grammar'
    More here : https://pythonprogramming.net/chunking-nltk-tutorial/ """
    words_to_dict = {}
    words_to_list=[]
    #file_name_ = str(file_name)+'.txt'
    #try :
    #    file  = open("files/"+file_name_,"r")
    #    DESIGN_TEXT = file.read()
    #except UnicodeDecodeError : 
    #   file  = open("files/"+file_name_,"r", encoding = "utf-8")
    #   DESIGN_TEXT = file.read() 
    #except Exception : 
    #    DESIGN_TEXT = file_name
    DESIGN_TEXT = text
    custom_sent_tokenizer = PunktSentenceTokenizer(DESIGN_TEXT)
    tokenized = custom_sent_tokenizer.tokenize(DESIGN_TEXT)
    sent_chunk =[]
    grammar = r"""Chunk :  {<NN.*><NN.*>}
                           {<VBG>}
                          
                          """
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            sent_chunk+=tagged
        cp=nltk.RegexpParser(grammar)
        print(cp.parse(sent_chunk))
        i=0
        for subtree in cp.parse(sent_chunk).subtrees(filter=lambda t: t.label() == 'Chunk' ):
            #print(' '.join([w for w, t in subtree.leaves()]))
            words_to_dict[i] = ' '.join([w for w, t in subtree.leaves()])
            words_to_list+=[' '.join([w for w, t in subtree.leaves()])]
            i+=1
    except Exception as e:
        print(str(e))
    return (words_to_list)


