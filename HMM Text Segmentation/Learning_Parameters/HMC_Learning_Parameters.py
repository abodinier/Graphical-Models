import numpy as np


def HMC_Parameters_Dict(dataset):
        
    Pi = {}
    A = {}
    B = {}
        
    for Z in dataset:
        
        Z0 = Z[0]
        X0 = Z0[0]
        Y0 = Z0[1]
        
        if X0 not in Pi.keys():
            Pi[X0] = 0
            
        Pi[X0] = Pi[X0] + 1
        
        if X0 not in B.keys():
            B[X0] = {}
            
        if Y0 not in B[X0].keys():
            B[X0][Y0] = 0
        
        B[X0][Y0] = B[X0][Y0] + 1
        
        
        Zi_prev = Z0
        
        for Zi in Z[1:]:
            x = Zi[0]
            y = Zi[1]
            
            if x not in Pi.keys():
                Pi[x] = 0
            
            if Zi_prev[0] not in A.keys():
                A[Zi_prev[0]] = {}
                
            if x not in A[Zi_prev[0]].keys():
                A[Zi_prev[0]][x] = 0
                
            if x not in B.keys():
                B[x] = {}
                
            if y not in B[x].keys():
                B[x][y] = 0
                            
            Pi[x] = Pi[x] + 1
            A[Zi_prev[0]][x] = A[Zi_prev[0]][x] + 1
            B[x][y] = B[x][y] + 1
            
            Zi_prev = Zi
            
    sum_Pi = np.sum(list(Pi.values()))
    for key in Pi.keys():
        Pi[key] = Pi[key]/sum_Pi
        
    for key_A_1 in A.keys():
        sum_A = np.sum(list(A[key_A_1].values()))
        for key_A_2 in A[key_A_1].keys():
            A[key_A_1][key_A_2] = A[key_A_1][key_A_2]/sum_A
            
    for key_B_1 in B.keys():
        sum_B = np.sum(list(B[key_B_1].values()))
        for key_B_2 in B[key_B_1].keys():
            B[key_B_1][key_B_2] = B[key_B_1][key_B_2]/sum_B
    
    
    return Pi, A, B