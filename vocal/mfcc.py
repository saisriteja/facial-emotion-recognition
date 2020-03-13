import argparse
import os
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import pickle
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-name","--folder_name",required =True,help="path for mfcc files")
args = vars(ap.parse_args())

file_folder = args['folder_name']

def read(file):
    (rate,sig) = wav.read(file)
    mfcc_feat = mfcc(sig,rate)
    return(mfcc_feat)

labels = ['a','d','su','sa','n','h']

def get_category(f):
    if f[0] == 's' and f[0:2] != 'su' and f[0:2] != 'sa':
        return(0)
        
    elif f[0]== 'a':
        return(1)
        
    elif f[0] == 'd':
        return(2)
        
    elif f[0] == 'n':
        return(3)
        
    elif f[0] == 'h':
        return(4)
    
    elif f[0] == 'f':
        return(7)
        
    elif f[0:2] == 'su':
        return(5)
        
    elif f[0:2] == 'sa':
        return(6)

files = os.listdir(file_folder)

data = []

for i in tqdm(files):
    
    path = file_folder+'/'+i
    #print(i,path)
    d = read(path)
    l = get_category(i)
    for j in d:
        data.append([j,l])
    #break

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

save_obj(data,file_folder)
