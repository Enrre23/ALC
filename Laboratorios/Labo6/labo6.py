import numpy as np

def metpot2k(A,tol=1e-15,K=1000):
    v = np.random.randint(1, 10, A.shape[1]) 
    v = np.array(v, dtype=float)

    v_moñito = trans_fak(v,2, A)
    e = productoM(np.array([v_moñito], dtype=float), traspuesta(np.array([v])))
    e = e[0][0]
    k = 0

    while(np.abs(e - 1) > tol and k < K):
        v = v_moñito
        v_moñito = trans_fak(v, 2, A)
        e = productoM(np.array([v_moñito], dtype=float), traspuesta(np.array([v])))
        e = e[0][0]
        k += 1

        
    lam = productoM(np.array([v_moñito], dtype=float), productoM(A, traspuesta(np.array([v_moñito], dtype=float))))
    lam = lam[0][0]
    epsilon = e - 1

    return v_moñito, lam, k, epsilon
    
def diagRH(A,tol=1e-15,K=1000):
    v1, lam_uno, _, _ = metpot2k(A, tol, K)
    e1 = crear_canonico(0, v1.size)
    H_v1 = matriz_de_reflexion((e1 - v1)/norma_2(e1 - v1), v1.size, 0)
    
    if(A.shape[0] == 2):
        S = H_v1
        D = productoM(S, A)
        D = productoM(D, traspuesta(S))

        return S, D
    
    else:
        B = productoM(H_v1, A)
        B = productoM(B, traspuesta(H_v1))
        A_moñito =  B[1: A.shape[0], 1: A.shape[0]]
        S_moñito, D_moñito = diagRH(A_moñito, tol, K)
        
        I = matriz_identidad(D_moñito.shape[0] + 1)
        submatriz_izq = I[1: I.shape[0], 0:1]
        submatriz_up = I[0: 1, 0: I.shape[1]]
        
        S = np.concatenate((submatriz_izq, S_moñito), axis = 1)
        S = np.concatenate((submatriz_up, S), axis = 0)
        S = productoM(H_v1, S)
        
        submatriz_up[0][0] = lam_uno
        D = np.concatenate((submatriz_izq, D_moñito), axis = 1)
        D = np.concatenate((submatriz_up, D), axis = 0)

        
        return S, D



def matriz_identidad(n):
    res = matrizNulaV2(n, n)
    #res = np.array(res, dtype=float)

    i = 0

    while(i < n):
        res[i][i] = 1
        i+= 1

    return res

def crear_canonico(n, c_fil):
    i = 0
    canonico = []
    
    while(i < c_fil):
        if(i == n):
            canonico.append(1)
        else:
            canonico.append(0)
        
        i+= 1

    canonico = np.array(canonico, dtype = float)

    return canonico

def matriz_de_reflexion(u, m, k):
    I = matriz_identidad(m-k)
    u = np.array([u], dtype = float)
    u_t = traspuesta(u)
    u_m = (-2)*productoM(u_t, u)
    Hk_moño = I + u_m

    return Hk_moño

def trans_fak(v, k, A):
    j = 0
    v_moñito = np.array(v.copy(), dtype=float)

    while(j < k):
        v_moñito = np.array([v_moñito])
        v_moñito = traspuesta(v_moñito)
        v_moñito = productoM(A, v_moñito)
        v_moñito = traspuesta(v_moñito)
        v_moñito = v_moñito[0]
        v_moñito = v_moñito/norma_2(v_moñito)
        j+= 1

    return v_moñito

def norma_2(v):
    res = 0
    i = 0

    while(i < v.size): 
        res = res + (v[i])**2 
        i += 1; 
        
    return np.sqrt(res) 

def matrizNulaV2(m, n):
   res = []
   i = 0
   
   while(i < m):
      res.append([])
      i += 1
      
   i = 0
    
   while(i < len(res)):
      j = 0
      while(j < n):
          res[i].append(0)
          j+= 1
       
      i += 1
     
   res = np.array(res, dtype=float)
   
   return res

def productoM(A, B):
     
    res = matrizNulaV2(A.shape[0], B.shape[1])
    transB = traspuesta(B)
    i = 0
     
    while(i < A.shape[0]):
       
        j = 0
         
        while(j < transB.shape[0]):
            k = 0
            acumulador = 0
             
            while(k < A[i].size):
                acumulador += A[i][k]*transB[j][k]
                k += 1
            
            res[i][j] = acumulador
            
            j+= 1
        
        i += 1
       
    return res

def traspuesta(A):
    i = 0
    t = []
    
    while(i < A.shape[1]):
        t.append([])
        i += 1

    for arreglo in A:
        j = 0
        while(j < arreglo.size):
            t[j].append(arreglo[j])
            j += 1
        
    t = np.array(t, dtype = float) 
    return t

A = np.array([[2,0,0], [0,1,0],[0,0,3]])

