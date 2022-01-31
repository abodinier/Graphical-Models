import numpy as np

class HMC_Forward_Backward_Restorator:
    
    def __init__(self, hmc = False):
        self.hmc = hmc
        
    def forward_prime(self, Omega_X, Y, parameter_set, default = 0):
        T = len(Y)
        alpha = np.zeros((T, len(Omega_X)))
        
        Pi = parameter_set["Pi"]
        A = parameter_set["A"]
        B = parameter_set["B"]

        alpha[0] = self.compute_alpha_1(Omega_X, Y, Pi, B, default)
        
        for t in range(0, T - 1):
            alpha[t + 1] = self.compute_alpha_t_plus_1(Omega_X, Y, t + 1, A, B, alpha[t], default)
     
        return alpha
    
    
    def backward_prime(self, Omega_X, Y, parameter_set, default = 0):
        T = len(Y)
        beta = np.zeros((T, len(Omega_X)))
        
        A = parameter_set["A"]
        B = parameter_set["B"]
        
        beta[T - 1] = self.compute_beta_T(Omega_X)
        
        for t in range(T - 2, -1, -1):
            beta[t] = self.compute_beta_t(Omega_X, Y, t + 1, A, B, beta[t + 1], default)
        
        return beta
    
    
    def compute_gamma(self, alpha, beta):
        gamma = alpha * beta
    
        gamma = np.transpose(gamma)/np.sum(gamma, axis = 1)
        gamma = np.transpose(gamma)
    
        return gamma
    
    def return_gamma(self, Omega_X, Y, parameter_set, epsilon_laplace = 0):
        alpha = self.forward_prime(Omega_X, Y, parameter_set, epsilon_laplace)
        beta = self.backward_prime(Omega_X, Y, parameter_set, epsilon_laplace)  
        
        gamma = self.compute_gamma(alpha, beta)
        
        return gamma
    
    def restore_X(self, Omega_X, Y, parameter_set, epsilon_laplace = 0):
        alpha = self.forward_prime(Omega_X, Y, parameter_set, epsilon_laplace)
        beta = self.backward_prime(Omega_X, Y, parameter_set, epsilon_laplace)  
        
        gamma = self.compute_gamma(alpha, beta)
        
        X = []
        indexes = np.argmax(gamma, axis = 1)
        
        for index in indexes:
            X.append(Omega_X[index])
        
        return X
    
    
    #######################
    ### COMPUTING ALPHA ###
    #######################
    
    def compute_alpha_1(self, Omega_X, Y, Pi, B, default = 0):
        alpha_1 = np.zeros(len(Omega_X))
        for idi, i in enumerate(Omega_X):
            if Y[0] in B[i]:
                # to complete
                alpha_1[idi]=Pi[i]*B[i][Y[0]]
        if np.sum(alpha_1) == 0:
            for idi, i in enumerate(Omega_X):
                # to complete
                alpha_1[idi]=Pi[i]
                #alpha_1[idi]=0
        alpha_1 = (alpha_1 + default)/np.sum(alpha_1 + default)
        return alpha_1
    
    def compute_alpha_t_plus_1(self, Omega_X, Y, t_plus_1, A, B, alpha_t, default = 0):
        alpha_t_plus_1 = np.zeros(len(Omega_X))
        yt_plus_1 = Y[t_plus_1]   
        for idi, i in enumerate(Omega_X):
            for idj, j in enumerate(Omega_X):
                if j in A and i in B:
                    if i in A[j] and yt_plus_1 in B[i]:
                        # to complete
                        alpha_t_plus_1[idi]+=B[i][yt_plus_1]*alpha_t[idj]*A[j][i]        
        
        if np.sum(alpha_t_plus_1) == 0:
            for idi, i in enumerate(Omega_X):
                for idj, j in enumerate(Omega_X):
                    if j in A:
                        if i in A[j]:
                            # to complete
                            alpha_t_plus_1[idi]+=alpha_t[idj]*A[j][i]
                            #alpha_t_plus_1[idi]+=0
        alpha_t_plus_1 = (alpha_t_plus_1 + default)/np.sum(alpha_t_plus_1 + default)
        return alpha_t_plus_1
    
    
    ######################
    ### COMPUTING BETA ###
    ######################
    
    def compute_beta_T(self, Omega_X):
        # to complete
        beta_T = np.ones(len(Omega_X))
        beta_T = beta_T / np.sum(beta_T)
        return beta_T
    
    
    def compute_beta_t(self, Omega_X, Y, t_plus_1, A, B, beta_t_plus_1, default = 0):
        beta_t = np.zeros(len(Omega_X))
        
        yt_plus_1 = Y[t_plus_1]
        
        for idi, i in enumerate(Omega_X):
            for idj, j in enumerate(Omega_X):
                # to complete
                if i in A and j in B:
                    if j in A[i] and yt_plus_1 in B[j]:
                        beta_t[idi]+=beta_t_plus_1[idj]*A[i][j]*B[j][yt_plus_1]   
                            
        if np.sum(beta_t) == 0:
            for idi, i in enumerate(Omega_X):
                for idj, j in enumerate(Omega_X):
                    # to complete
                    if i in A :
                        if j in A[i]:
                            beta_t[idi]+=beta_t_plus_1[idj]*A[i][j]   
                            #beta_t[idi]+=0
        
        beta_t = (beta_t + default)/np.sum(beta_t + default)
        return beta_t
