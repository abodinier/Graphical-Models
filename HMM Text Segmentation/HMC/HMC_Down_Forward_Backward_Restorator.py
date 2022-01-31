import numpy as np

class HMC_Down_Forward_Backward_Restorator:
    
    def __init__(self, hmc = False):
        self.hmc = hmc
        
    def forward_prime(self, Omega_X, list_Y, parameter_set, default = 0):
        T = len(list_Y[0])
        alpha = np.zeros((T, len(Omega_X)))
        
        Pi_hmc = parameter_set["Pi_HMC"]
        A_hmc = parameter_set["A_HMC"]
        list_B_hmc = parameter_set["list_B_HMC"]

        alpha[0] = self.compute_alpha_1(Omega_X, list_Y, Pi_hmc, list_B_hmc, default)
        
        for t in range(0, T - 1):
            alpha[t + 1] = self.compute_alpha_t_plus_1(Omega_X, list_Y, t + 1, A_hmc, list_B_hmc, alpha[t], default)
     
        return alpha
    
    
    def backward_prime(self, Omega_X, list_Y, parameter_set, default = 0):
        T = len(list_Y[0])
        beta = np.zeros((T, len(Omega_X)))
        
        A_hmc = parameter_set["A_HMC"]
        list_B_hmc = parameter_set["list_B_HMC"]
        
        beta[T - 1] = self.compute_beta_T(Omega_X)
        
        for t in range(T - 2, -1, -1):
            beta[t] = self.compute_beta_t(Omega_X, list_Y, t + 1, A_hmc, list_B_hmc, beta[t + 1], default)
        
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
    
    def compute_alpha_1(self, Omega_X, list_Y, Pi_hmc, list_B_hmc, default = 0):
        alpha_1 = np.zeros(len(Omega_X))
        for idi, i in enumerate(Omega_X):
            if list_Y[0][0] in list_B_hmc[0][i]: # Si le mot est connu, on ne change rien par rapport au fwd-bwd classique
                # to complete
                alpha_1[idi]=Pi_hmc[i]*list_B_hmc[0][i][list_Y[0][0]]

        if np.sum(alpha_1) == 0: # Si le mot list_Y[i] est inconnu mais qu'on a  déjà croisé cette structure :
            for idi, i in enumerate(Omega_X):
                if list_Y[1][0] in list_B_hmc[1][i] :
                    # to complete
                    alpha_1[idi]=Pi_hmc[i]*list_B_hmc[1][i][list_Y[1][0]]

        if np.sum(alpha_1) == 0: # Si on n'a jamais croisé cette structure non plus :
            for idi, i in enumerate(Omega_X):
                # to complete
                alpha_1[idi]=Pi_hmc[i]

        alpha_1 = (alpha_1 + default)/np.sum(alpha_1 + default)
        return alpha_1
    
    def compute_alpha_t_plus_1(self, Omega_X, list_Y, t_plus_1, A_hmc, list_B_hmc, alpha_t, default = 0):
        alpha_t_plus_1 = np.zeros(len(Omega_X))
        yt_plus_1, y3t_plus_1 = list_Y[0][t_plus_1], list_Y[1][t_plus_1]  

        for idi, i in enumerate(Omega_X):
            for idj, j in enumerate(Omega_X):
                if j in A_hmc and i in list_B_hmc[0]: #Si on a déjà croisé y :
                    if i in A_hmc[j] and yt_plus_1 in list_B_hmc[0][i]:
                        # to complete
                        alpha_t_plus_1[idi]+=list_B_hmc[0][i][yt_plus_1]*alpha_t[idj]*A_hmc[j][i]        
        
        if np.sum(alpha_t_plus_1) == 0:
            for idi, i in enumerate(Omega_X):
                for idj, j in enumerate(Omega_X):
                    if j in A_hmc and i in list_B_hmc[1]: #Si on a jamais croisé y, mais croisé cette structure :
                        if i in A_hmc[j] and y3t_plus_1 in list_B_hmc[1][i]:
                            # to complete
                            alpha_t_plus_1[idi]+=list_B_hmc[1][i][y3t_plus_1]*alpha_t[idj]*A_hmc[j][i]

        if np.sum(alpha_t_plus_1) == 0: # Si on n'a jamais croisé ni l'un ni l'autre     
            for idi, i in enumerate(Omega_X):
                for idj, j in enumerate(Omega_X):
                    if j in A_hmc:
                        if i in A_hmc[j]:
                            # to complete
                            alpha_t_plus_1[idi]+=alpha_t[idj]*A_hmc[j][i]

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
    
    
    def compute_beta_t(self, Omega_X, list_Y, t_plus_1, A_hmc, list_B_hmc, beta_t_plus_1, default = 0):
        beta_t = np.zeros(len(Omega_X))
        # to complete

        yt_plus_1, y3t_plus_1 = list_Y[0][t_plus_1], list_Y[1][t_plus_1]
        
        for idi, i in enumerate(Omega_X): 
            for idj, j in enumerate(Omega_X):
                # to complete
                if i in A_hmc and j in list_B_hmc[0]: # Si on a déjà croisé ce mot
                    if j in A_hmc[i] and yt_plus_1 in list_B_hmc[0][j]:
                        beta_t[idi]+=beta_t_plus_1[idj]*A_hmc[i][j]*list_B_hmc[0][j][yt_plus_1]   
                            
        if np.sum(beta_t) == 0:
            for idi, i in enumerate(Omega_X): 
                for idj, j in enumerate(Omega_X):
                    # to complete
                    if i in A_hmc and j in list_B_hmc[1]: # Si on n'a jamais croisé ce mot mais déjà croisé cette structure :
                        if j in A_hmc[i] and y3t_plus_1 in list_B_hmc[1][j]:
                            beta_t[idi]+=beta_t_plus_1[idj]*A_hmc[i][j]*list_B_hmc[1][j][y3t_plus_1]

        if np.sum(beta_t) == 0:
            for idi, i in enumerate(Omega_X):
                for idj, j in enumerate(Omega_X):
                    # to complete
                    if i in A_hmc :
                        if j in A_hmc[i]:
                            beta_t[idi]+=beta_t_plus_1[idj]*A_hmc[i][j]   
                            #beta_t[idi]+=0

        beta_t = (beta_t + default)/np.sum(beta_t + default)
        return beta_t
        