####################
### DEPENDENCIES ###
####################

import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("/Users/Alex2/Cours/2A/MA202/TP2/")
f = open("results_down.txt", "a")
sys.stdout = f
from HMC.HMC_Down_Forward_Backward_Restorator import HMC_Down_Forward_Backward_Restorator
from Learning_Parameters.HMC_Learning_Parameters import HMC_Parameters_Dict
from Dataset.CoNLL2000.Load_CoNLL2000 import load_conll2000
import numpy as np
import time


#######################
### LOADING DATASET ###
#######################

Omega_X = ['INTJ', 'DET', 'CCONJ', 'NOUN', 'SYM', 'PART', 'ADP', 'ADV', 'PRON', 'X', 'VERB', 'NUM', 'PROPN', 'PUNCT', 'ADJ']
train_set, test_set = load_conll2000(path = "/Users/Alex2/Cours/2A/MA202/TP2/Dataset/CoNLL2000/")

Omega_X_dict = dict(zip(Omega_X, np.arange(0, len(Omega_X))))
vocabulary = list(set([y for sent in train_set for x, y in sent]))

vocabulary_dict = dict.fromkeys(vocabulary, 0)
for sent in train_set:
    for taf, word in sent:
        vocabulary_dict[word] = vocabulary_dict[word] + 1
    
    
########################
### USEFUL FUNCTIONS ###
########################
    
def convert_word_to_features(idw, word, suffix_len):
    features = word[-suffix_len:]
    
    if word[0] == word[0].upper():
        features = features + "U"
    else:
        features = features + "NU"
        
    if "-" in word:
        features = features + "H"
    else:
        features = features + "NH"
        
    if idw == 0:
        features = features + "F"
    else:
        features = features + "NF"
        
    if True in [c.isdigit() for c in word]:
        features = features + "D"
    else:
        features = features + "ND"
        
    return features

def construct_train_set_features(train_set, suffix_len):
    train_set_features = []
    for sent in train_set:
        sent_features = [(x, convert_word_to_features(idy, y, suffix_len)) for idy, (x, y) in enumerate(sent)]
        train_set_features.append(sent_features)
        
    return train_set_features


###############################
### TRAIN SET CONSTRUCTIONS ###
###############################
    
train_set_3 = construct_train_set_features(train_set, 3)
train_set_2 = construct_train_set_features(train_set, 2)
train_set_1 = construct_train_set_features(train_set, 1)
train_set_0 = construct_train_set_features(train_set, 0)



##################################
### STATIONNARY HMC PARAMETERS ###
##################################

start_time = time.time()
Pi, A, B3 = HMC_Parameters_Dict(train_set_3)
Pi, A, B2 = HMC_Parameters_Dict(train_set_2)
Pi, A, B1 = HMC_Parameters_Dict(train_set_1)
Pi, A, B0 = HMC_Parameters_Dict(train_set_0)
Pi, A, B = HMC_Parameters_Dict(train_set)
training_time = time.time() - start_time 

list_B = [B, B3, B2, B1, B0]


#####################
### PARAMETER SET ###
#####################

parameter_set = {
    "Pi_HMC": Pi,
    "A_HMC": A,
    "list_B_HMC": list_B
}


###############
### TESTING ###
###############


hmc_res = HMC_Down_Forward_Backward_Restorator()

score, total = 0, 0

score_kw, total_kw = 0, 0
score_uw, total_uw = 0, 0

default = 10**(-10)

for ids, sent in enumerate(test_set):
    
    X = [x for x, y in sent]
    Y = [y for x, y in sent]
    
    Y3 = [convert_word_to_features(idy, y, 3) for idy, (x, y) in enumerate((sent))]
    Y2 = [convert_word_to_features(idy, y, 2) for idy, (x, y) in enumerate((sent))]
    Y1 = [convert_word_to_features(idy, y, 1) for idy, (x, y) in enumerate((sent))]
    Y0 = [convert_word_to_features(idy, y, 0) for idy, (x, y) in enumerate((sent))]
    
    list_Y = [Y, Y3, Y2, Y1, Y0]
    
    X_hat = hmc_res.restore_X(Omega_X, list_Y, parameter_set, default)
    
    for t in range(len(Y)):
        score += 1 * (X[t] == X_hat[t])
            
        if Y[t] in vocabulary_dict:
            total_kw += 1
            score_kw += 1 * (X[t] == X_hat[t])
            
        else:
            total_uw += 1
            score_uw += 1 * (X[t] == X_hat[t])
            
        total += 1
        
    if ids % 100 == 0 and ids != 0:
        print(ids, score/total * 100)
    
print("HMC Accuracy on CoNLL 2000 POS Tagging:", 100 - score/total * 100, "%")

print("KW:", 100 - score_kw/total_kw * 100)
print("UW:", 100 - score_uw/total_uw * 100)

print("Training time:", training_time)
print('\n\n')
f.close()