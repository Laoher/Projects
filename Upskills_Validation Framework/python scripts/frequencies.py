
import pandas as pd
from text_processing import txt_processing
import json 
import time
class jsdict(dict):
        def __str__(self):
            return json.dumps(self)
def add_file_frequencies (file_name) :
    """ Old version of the function process_file() from the file test functional.py. The json.txt file is used in case we want to add a text to an existing set of text without rrecalculating the frequencies of the previous texts"""
    start_time = time.time()
    with open("files/json_frequencies.txt", "r", encoding = "utf-8-sig") as file:
        json_str = file.read()
    dic_freq = json.loads(json_str)

    try :
        df=pnd.read_csv('files/frequencies3.csv');
        x=df['file name'].values.tolist()
    except FileNotFoundError :
        x=[]
    
    if ("file name" in dic_freq.keys()) :
        dic_freq["file name"] += [file_name]
    else :
        dic_freq["file name"]= [file_name]
    nbr_txt = 0
    nbr_txt = len(dic_freq["file name"])
    try : 
        freq = txt_processing(file_name)
    except : #au cas ou on a rentre que le texte
        freq = FreqDist(file_name)
        
    for w in freq :
        if (w[0] in dic_freq.keys()) :
     
            if (len(dic_freq[w[0]])==nbr_txt) :
                dic_freq[w[0]] += [w[1]]
                
            else : 
                dic_freq[w[0]]+=[0]*(nbr_txt-len(dic_freq[w[0]])-1)
                dic_freq[w[0]]+=[w[1]]
            
        else :
            dic_freq[w[0]] = [0]*(nbr_txt-1)        
            dic_freq[w[0]]+=[w[1]]
    for w in dic_freq.keys() :
        l=len(dic_freq[w])
        if (l!=nbr_txt) :
            dic_freq[w]+=[0]*(nbr_txt-l)
            
    with open("files/json_frequencies.txt", "w", encoding = "utf_8") as file:
        file.write(str(jsdict(dic_freq)))
           
   
    print("Temps d execution : %s secondes ---" % (time.time() - start_time))
    return(dic_freq)
     

    