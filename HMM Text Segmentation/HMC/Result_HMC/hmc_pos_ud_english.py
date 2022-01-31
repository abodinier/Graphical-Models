####################
### DEPENDENCIES ###
####################

import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("/Users/Alex2/Cours/2A/MA202/TP2/")
from HMC.HMC_Forward_Backward_Restorator import HMC_Forward_Backward_Restorator
from Learning_Parameters.HMC_Learning_Parameters import HMC_Parameters_Dict
from Dataset.UD_English.Load_UD_English import load_ud_english
import numpy as np
import time
sys.stdout = open("results.txt", "a")


#######################
### LOADING DATASET ###
#######################

train_set, test_set = load_ud_english(path = "/Users/Alex2/Cours/2A/MA202/TP2/Dataset/UD_English/")
Omega_X = list(set([x for sent in train_set for x, y in sent]))

Omega_X_dict = dict(zip(Omega_X, np.arange(0, len(Omega_X))))
vocabulary = list(set([y for sent in train_set for x, y in sent]))

vocabulary_dict = {}
for idw, word in enumerate(vocabulary):
    vocabulary_dict[word] = idw
    

##################################
### STATIONNARY HMC PARAMETERS ###
##################################

start_time = time.time()
Pi, A, B = HMC_Parameters_Dict(train_set)
training_time = time.time() - start_time 


#####################
### PARAMETER SET ###
#####################

parameter_set = {
    "Pi": Pi,
    "A": A,
    "B": B
}


###############
### TESTING ###
###############

hmc_res = HMC_Forward_Backward_Restorator()

score, total = 0, 0

score_kw, total_kw = 0, 0
score_uw, total_uw = 0, 0

default = 10**(-10)

for ids, sent in enumerate(test_set):
    
    X = [x for x, y in sent]
    Y = [y for x, y in sent]
    
    X_hat = hmc_res.restore_X(Omega_X, Y, parameter_set, default)
    
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
    
print("HMC Accuracy on ud english POS Tagging:", 100 - score/total * 100, "%")

print("KW:", 100 - score_kw/total_kw * 100)
print("UW:", 100 - score_uw/total_uw * 100)
print("total known words : %d" %(total_kw))
print("total unknown words : %d" %(total_uw))

print("Training time: ", training_time)
print("\n\n")
sys.stdout.close()