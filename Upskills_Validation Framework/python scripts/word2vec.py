# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:37:21 2018

@author: BadrBelkeziz
"""
import gensim
from gensim import corpora, models, similarities
import pandas as pd
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
import nltk
from nltk.tokenize import PunktSentenceTokenizer
import numpy as np
import gc
import text_processing
from nltk.corpus import stopwords
import itertools 
from process_text import text_tokenization

stop_words =set(stopwords.words('english')+ [',','.',';',':','’','&','"','-','>','<','*','(',')','-','+','?','',"''",'!','``',"''",'»','ï','¿1', '--','i','/i'])

def guess_words(texts_path, output_name, model_name):
    global word_model
    if ("word_model" not in globals()) :
        word_model = gensim.models.Word2Vec.load(model_name)

    grammar = r"""negativ :  <RB><.*>?{<.*>}
    positiv :  {<.*>} """

    df=pd.read_csv(texts_path);
    columns_file = df.columns.values.tolist()
    df_dict = df.to_dict()
    data_dict = {}
    texts = df_dict[columns_file[2]] 
    theme = df_dict[columns_file[1]]
    words_dict = {}
    for l in range(len(texts)) :
            words_dict["text " + str(l)] = []
            negativ=[]
            positiv =[]
            DESIGN_TEXT = texts[l]
            custom_sent_tokenizer = PunktSentenceTokenizer(DESIGN_TEXT)
            tokenized = custom_sent_tokenizer.tokenize(DESIGN_TEXT)
            sent_chunk =[]
            try:
                for i in tokenized:
                    words = nltk.word_tokenize(i)
                    tagged = nltk.pos_tag(words)
                    sent_chunk+=tagged
                cp=nltk.RegexpParser(grammar)
        #print(cp.parse(sent_chunk))
                for subtree in cp.parse(sent_chunk).subtrees(filter=lambda t: t.label() == 'negativ' ):
                    #print(' '.join([w for w, t in subtree.leaves()]))
                    negativ+=[w for w, t in subtree.leaves()]
                for subtree in cp.parse(sent_chunk).subtrees(filter=lambda t: t.label() == 'positiv' ):
                    #print(' '.join([w for w, t in subtree.leaves()]))
                    positiv+=[w for w, t in subtree.leaves()]
    
            except Exception as e:
                print(str(e)) 
            #print(positiv)
            positiv =  [w for w in positiv if not w in stop_words]
            positive = []
            negativ =  [w for w in negativ if not w in stop_words]
            negative = []
            #print(positiv)
            for i in positiv :
                try :
                    word_model.wv.most_similar(i)
                    positive +=[i]
                except KeyError :
                    print(i + " Not in vocabulary")
            for i in negativ :
                try :
                    word_model.wv.most_similar(i)
                    negative +=[i]
                except KeyError :
                    print(i + " Not in vocabulary")
            #print(positive)
        
            answers = word_model.wv.most_similar(positive=positive,negative=negative)
            for word in answers :
                percent = round(word[1]*100)
                words_dict["text " + str(l)]+=[word[0] + " " + str(round(percent))]
    df = pd.DataFrame(words_dict)
    df.T.to_csv(output_name + ".csv")


def generator_wt (file_name) : 
    """ Returns a generator (instead of a list, to save some memory) of tokenized words from the file. In order to improve the, you can try and imrpove the text processing (select more or less words, groups of words, apply stemming or other NLP techniques ... """
    yield_wt = itertools.chain()
    with open(file_name, encoding="utf-8") as fh:
        for line in fh:
            sent_token = sent_tokenize(line)
            if (sent_token!=[]):
                yield_wt = itertools.chain(yield_wt,text_tokenization(sent_token[0], automatic_group = False))
    return(yield_wt)
           

def create_words_model(text_file_name, model_name,min_count=3, workers=4, size = 100, window=50) :
    """ Uses the gensim library to create a word to vec model and save it. Find more on the gensim doc """ 
    word_token=generator_wt(text_file_name)
    #file  = open("wki.txt","r",encoding="utf-8")
    #DESIGN_TEXT = file.read()
    word_token =[[ w for w in sent if not w in stop_words] for sent in word_token]
    model = gensim.models.Word2Vec(word_token, min_count=min_count, workers=workers, size = size, window=window)
    model.save(model_name)
                
            
def update_model(file_name,model_name) :
    """ Allows you to update your word to vec model with a new file """ 
    model = gensim.models.Word2Vec.load(model_name,mmap='r')
    word_token=generator_wt(file_name)
    word_token =[[ w for w in sent ] for sent in word_token]
    model.build_vocab(word_token,update=True)
    model.train(word_token, len(word_token), epochs = model.epochs)
    model.save(model.name)
    gc.collect()
    print("%s added to the model " % file_name)        

        

