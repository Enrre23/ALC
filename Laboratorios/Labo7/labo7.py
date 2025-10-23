import numpy as np
from labo6 import diagRH

def transiciones_al_azar_continuas(n):
    res = []
    i = 0

    while(i < n):
        j = 0
        fila = []
        while(j < n):
            fila.append(np.random.uniform(0, 1))
            j += 1

        res.append(fila)
        i+= 1

    res = np.array(res)
    
    fil_acutal = 0
    while(fil_acutal < res.shape[0]):
        col_actual = 0
        acumulador = 0
        while(col_actual < res.shape[1]):
            acumulador = acumulador + res[fil_acutal][col_actual]
            col_actual = col_actual + 1
            
        col_actual = 0

        while(col_actual < res.shape[0]):
            res[fil_acutal][col_actual] = res[fil_acutal][col_actual]/acumulador
            col_actual += 1 

        fil_acutal+= 1


    return traspuesta(res)   
         
        
def transiciones_al_azar_uniformes(n,thres):
    res = []
    i = 0

    while(i < n):
        j = 0
        fila = []
        while(j < n):
            num_random = np.random.uniform(0, 1)

            if(num_random < thres):
                fila.append(1)
            if(num_random > thres):
                fila.append(0)
            else:
                fila.append(num_random)
                
            j += 1

        res.append(fila)
        i+= 1

    res = np.array(res)
    
    fil_acutal = 0  
    while(fil_acutal < res.shape[0]):
        col_actual = 0
        acumulador = 0
        while(col_actual < res.shape[1]):
            acumulador = acumulador + res[fil_acutal][col_actual]
            col_actual = col_actual + 1
            
        col_actual = 0

        while(col_actual < res.shape[0]):
            res[fil_acutal][col_actual] = res[fil_acutal][col_actual]/acumulador
            col_actual += 1 

        fil_acutal+= 1


    return traspuesta(res)  


def nucleo(A,tol=1e-15):
    S, D = diagRH(productoM(traspuesta(A), A))
    i = 0
    paso = True
    res = np.array([])
    while(i < A.shape[0]):
        if(abs(0 - D[i][i]) <= tol):
            if(paso):
                res = S[0: S.shape[0], i: i + 1]
                paso = False   
            else:
                col = S[0: S.shape[0], i:i + 1]
                res = np.concatenate((res, col), axis = 1)

        i = i + 1
    
    return res

def crea_rala(listado,m_filas,n_columnas,tol=1e-15):
    i = 0
    lista_Aij = listado[2]
    indices_i = listado[0]
    indices_j = listado[1]
    diccionario = dict()

    while(i < len(lista_Aij)):
        if(abs(lista_Aij[i]) >= tol):
            diccionario[(indices_i[i], indices_j[i])] = lista_Aij[i]
        
        i = i + 1

    return [diccionario, (m_filas, n_columnas)]

def multiplica_rala_vector(A,v):
    c_filas = A[1][0]
    c_columnas = A[1][1]
    dict = A[0]
    i = 0
    res = []
    
    while(i < c_filas):
        j = 0
        acumulador = 0

        while(j < c_columnas):
            if((i,j) in dict):
                acumulador = acumulador + dict[(i,j)]*v[j]
            
            j+= 1

        i+= 1

        res.append(acumulador)
    
    return np.array(res, dtype = float)
    


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

