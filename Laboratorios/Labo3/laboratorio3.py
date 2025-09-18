import numpy as np
import random
import matplotlib.pyplot as plt

#Parte 1)
def norma(x, p):
    i = 0
    res = 0

    if(p == 'inf'):
        maximo = 0
        j = 0

        while(j < x.size):
            componente_abs =  np.float64(np.abs(np.float64(x[j])))
            if(componente_abs > maximo):
                maximo = componente_abs
            
            j+= 1

        return maximo

    while(i < x.size):
        res += np.float64(np.abs(np.float64(x[i]))**p)          #aqui
        i+= 1

    return np.float64(res**(1/p))


def normaliza(X, p):        #Normalizar es dividir al vector por su norma para conseguir un vector unitario
    i = 0
    Y = []

    while(i < len(X)):
        normalizado = X[i]*(1/norma(X[i], p))
        Y.append(normalizado)
        i+= 1

    return Y

#Parte 2)

def normaMatMC(A, q, p, Np):            #Se debe retorna un tupla t. t[0] = norma y t[1] = vector
    vectores_random = vectores_alazar(A.shape[1], Np, p)        
    transformados = []
    norma_q_trans = []
    i = 0


    while(i < len(vectores_random)):
        transformados.append(A@vectores_random[i])
        i+= 1

    j = 0

    while(j < len(transformados)):
        norma_q_trans.append(norma(transformados[j], q))
        j+= 1

    k = 1
    posicion_max = 0
    max = norma_q_trans[0]

    while(k < len(norma_q_trans)):
        if(norma_q_trans[k] > max):
            max = norma_q_trans[k]
            posicion_max = k
        
        k+= 1

    
    res = (np.float64(norma_q_trans[posicion_max]), vectores_random[posicion_max])

    return res
    
def normaExacta(A, p):
    max = None
    
    if(p == 1):
        max = 0
        col_actual = 0
        
        while(col_actual < A.shape[1]):
            j = 0 
            acumulador = 0

            while(j < A.shape[0]):
                acumulador += np.abs(np.float64(A[j][col_actual]))
                j+= 1

            if(acumulador > max):
                max = acumulador

            col_actual += 1
    if(p == 'inf'):
        max = 0
        fil_actual = 0

        while(fil_actual < A.shape[0]):
            j = 0
            acumulador = 0

            while(j < A.shape[1]):
                acumulador += np.abs(np.float64(A[fil_actual][j]))
                j+= 1

            if(acumulador > max):
                max = acumulador

            fil_actual += 1

    return max


#Parte 3):
def condMC(A, p, cant_vec):
    norma_a = np.float64((normaMatMC(A, p, p, cant_vec))[0])
    inversa_a = inversa(A)
    norma_inversa_a = np.float64((normaMatMC(inversa_a, p, p, cant_vec))[0])
    num_cond_a = np.float64(norma_a*norma_inversa_a)

    return num_cond_a

def condExacta(A, p):
    norma_a = normaExacta(A, p)
    inversa_a = inversa(A)
    norma_inversa_a = normaExacta(inversa_a, p)
    num_cond_a = norma_a*norma_inversa_a
    
    return num_cond_a


#Funciones auxiliares
def vectores_alazar(cant_comp, cant_v, p):
    vectores = []
    i = 0

    while(i < cant_v): 
        vector = []
        j = 0

        while(j < cant_comp):
            vector.append(random.randint(1,1000000)) #aumentar esto
            j+= 1

        vectores.append(np.array(vector))
        i+= 1
    
    normalizados = normaliza(vectores, p)
    

    return normalizados

def inversa(A):
    copia = None
    existe_inversa = [True]
    res = None

    if(A.shape[0] == A.shape[1]):
        copia = copiar_matriz(A)
        identidad = matriz_identidad(A.shape[0])
        copia = np.concatenate((copia, identidad), axis=1)

        inversa_aux(copia, 0, A.shape[0] - 1, existe_inversa)

        if(existe_inversa[0]):
            inversa_aux_dos(copia, 1, A.shape[0] - 1, existe_inversa)
        
    if(existe_inversa[0]):
        res = inversa_aux_tres(copia, A.shape[0])
    return res                    #Tratar el caso de la matriz identidad 1x1 aparte
        

def inversa_aux(A, i, t, b):
    
    if(i == t):
        return 
    
    elem_diagonal = A[i][i]
    
    if(elem_diagonal != 0):
        n = i + 1

        while(n <= t):
            if(A[n][i] != 0):
                sumar_fila_multiplo(A, n, i, -A[n][i]/elem_diagonal)
            
            n+= 1
        
    if(elem_diagonal == 0):
        n = i + 1
        paso = True

        while(n <= t and paso):
            if(A[n][i] != 0):
                intercambiarFilas(A, n, i)
                paso = False

            n+= 1

        if(A[i][i] == 0):
            b[0] = None
            return 
        else:
            n = i + 1
            while(n <= t):
                if(A[n][i] != 0):
                    sumar_fila_multiplo(A, n, i, A[n][i]/A[i][i])
            
                n+= 1
    
    inversa_aux(A, i + 1, t, b)

def inversa_aux_dos(A, i , t, b): #A, i = 1, t = 1
    if(i == t + 1): # t = 2
        return 
    
    elem_diagonal = A[i][i]
    
    if(elem_diagonal != 0):
        n = i - 1

        while(n >= 0):
            if(A[n][i] != 0):
                sumar_fila_multiplo(A, n, i, -A[n][i]/elem_diagonal)
            
            n+= -1

    inversa_aux_dos(A, i + 1, t, b)


def inversa_aux_tres(A, f):
    i = 0
    while(i < f):
        elemento = 1/A[i][i]
        A[i] = A[i]*elemento
        i+= 1

   
    return A[0:f, f:2*f]
    


def intercambiarFilas(A, i, j):
    fila_guardada = copiar_vector(A[i])  #A[i].copy()
    A[i] = A[j]
    A[j] = fila_guardada

def sumar_fila_multiplo(A, i, j, s):
    fila_guardada = copiar_vector(A[j])  #A[j].copy()
    k = 0

    while(k < fila_guardada.size):
        fila_guardada[k] = fila_guardada[k]*s
        k += 1
    
    k = 0 

    while(k < A[i].size):
        A[i][k] = A[i][k] + fila_guardada[k]
        k += 1

def matriz_nula(m, n):
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


def matriz_identidad(n):
    res = matriz_nula(n, n)

    i = 0

    while(i < n):
        res[i][i] = 1
        i+= 1

    return res

def copiar_matriz(M):
    cant_de_fila = M.shape[0]
    i = 0
    res = []

    while(i < cant_de_fila):
        j = 0
        fila_nueva = []

        while(j < M[i].size): 
            fila_nueva.append(M[i][j])
            j+= 1
        
        res.append(fila_nueva)
        i+= 1

    
    res = np.array(res)

    return res

def copiar_vector(v):
    res = []
    i = 0

    while(i < v.size):
        res.append(v[i])
        i+= 1

    res = np.array(res)
    return res



    

"""
j = 0
vectores = []
while(j < 100000):
    vectores.append(np.array([random.randint(1,10), random.randint(1,10)]))
    j+= 1

j = 0
while(j < len(vectores)):
    vectores[j] = vectores[j]*(1/norma(vectores[j], 200))
    j+= 1

x = []
y = []

j = 0
while(j < len(vectores)):
    x.append(vectores[j][0])
    y.append(vectores[j][1])
    j+= 1
    
plt.plot(x, y)
plt.show()
"""


