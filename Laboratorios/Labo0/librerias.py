import numpy as np
def esCuadrada(arr):
    res = False

    if(cantDeCol(arr) == cantDeFilas(arr)):
        res = True

    return res

def triangSup(arr):

    if(cantDeCol(arr) == 1 and cantDeFilas(arr) == 1):
        return arr
    
    actual = 1
    posicion = 0
    u = matrizNula(arr)
    u = np.array(u)

    while(actual < cantDeCol(arr)):
        j = actual
        while(j < cantDeCol(arr)):
            u[posicion][j] = arr[posicion][j]
            j+= 1

        actual += 1
        posicion += 1

    return u

def triangInf(arr):

    if(cantDeCol(arr) == 1 and cantDeFilas(arr) == 1):
        return arr
    
    actual = 0
    posicion = 1
    res = matrizNula(arr)
    res = np.array(res)

    while(actual <= cantDeCol(arr)//2):
        j = 0
        while(j <= actual):
            res[posicion][j] = arr[posicion][j]
            j+= 1

        actual += 1
        posicion += 1

    return res

def diagonal(arr):
    i = 0
    res = matrizNula(arr)
    res = np.array(res)

    while(i < cantDeCol(arr)):
        res[i][i] = arr[i][i]
        i += 1

    return res

def traza(arr):
    i = 0
    res = 0
    while(i < cantDeCol(arr)):
        res += arr[i][i]
        i += 1

    return res

def traspuesta(arr):
    i = 0
    t = []
    
    while(i < cantDeCol(arr)):
        t.append([])
        i += 1

    for arreglo in arr:
        j = 0
        while(j < arreglo.size):
            t[j].append(arreglo[j])
            j += 1
        
    t = np.array(t) 
    return t


def esSimetrica(arr):
    res = False

    if(esCuadrada(arr)):
        trans = traspuesta(arr)
        res = True
        i = 0

        while(i < cantDeCol(arr) and res):
            j = 0
            while(j < arr[i].size and res):
                if(arr[i][j] != trans[i][j]):
                    res = False
                j += 1

            i += 1

    return res

def caclcularAx(A, x):
    return productoM(A,x)

def intercambiarFilas(A, i, j):
    fila_guardada = A[i].copy()
    A[i] = A[j]
    A[j] = fila_guardada

def sumar_fila_multiplo(A, i, j, s):
    copia = A[j].copy()
    k = 0

    while(k < copia.size):
        copia[k] = copia[k]*s
        k += 1
    
    k = 0 

    while(k < A[i].size):
        A[i][k] = A[i][k] + copia[k]
        k += 1
    

def esDiagonalmenteDominante(A):
    fila_actual = 0
    res = True
    while(fila_actual < cantDeFilas(A) and res):
        valor_diago = absoluto(A[fila_actual][fila_actual])
        acumulador = 0 
        j = 0

        while(j < A[fila_actual].size):
            if(fila_actual != j):
                acumulador += absoluto(A[fila_actual][j])

            j+= 1
        
        if(valor_diago <= acumulador):
            res = False

        fila_actual += 1

    return res

def matrizCirculante(v):
    permutacion =  v
    res = []
    res.append(v)
    i = 1

    while(i < v.size):
       permutacion = permutacionCiclica(permutacion)
       res.append(permutacion)
       i += 1
    
    res = np.array(res)
    return res

def matrizVandermonde(v):
    res = []
    i = 0

    while(i < v.size):
        fila_nueva = filaPotenciasIesima(v, i)
        res.append(fila_nueva)
        i+= 1

    res = np.array(res)
    return res

def matrizFibonacci(n):
        A = matrizNulaV2(n,n)
        i = 0
        
        while(i < cantDeFilas(A)):
            j = 0
            
            while (j < A[i].size):
                A[i][j] = sucesionFibo(i + j + 2)
                j += 1
                
            i += 1
            
        return A


def matrizHilbert(n):
        H = matrizNulaV2(n,n)
        i = 0
        
        while(i < cantDeFilas(H)):
            j = 0
            
            while (j < H[i].size):
                H[i][j] = 1/(i + j + 1 + 2)
                j += 1
                
            i += 1
            
        return H

def evaluar_polinomio(i, n):
    res = 0

    if(i == 0):
        
        m = productoM(np.array([[1, -1, 1, -1, 1, -1]]),np.array([[n**5],[n**4],[n**3],[n**2],[n],[1]]))
        res = m[0][0]


    if(i == 1):
        m = productoM(np.array([[1,3]]),np.array([[n**2],[1]]))
        res = m[0][0]


    if(i == 2):
        m = productoM(np.array([[1,-2]]),np.array([[n**10],[1]]))
        res = m[0][0]

    return res


def sucesionFibo(n):

    if(n == 0):
        return 0

    if(n == 1):
        return 1
     
    matriz_fibo = np.array([[1, 1],[1, 0]])

    i = 2
    
    f = np.array([[1,1],[1,0]])

    while(i < n):
        matriz_fibo = productoM(matriz_fibo, f)
        i+= 1

    return matriz_fibo[0][0]

def filaPotenciasIesima(v, i):
    w = v.copy()
    j = 0
    
    while(j < w.size):
        w[j] = w[j]**i
        j += 1

    return w
          

def permutacionCiclica(v):

    w = v.copy()
    ultimo = w[w.size - 1]
    guardado = w[0]
    i = 1
    
    while(i < w.size):
        x = w[i]
        w[i] = guardado
        i+= 1
        guardado = x

    w[0] = ultimo

    return w

def row_echelon(A):
    i = 0

    while(i < cantDeFilas(A) - 1):
        paso = True

        fila_actual = A[i]
        j = i + 1
        h = i
        while(h < cantDeFilas(A) and paso and fila_actual[i] == 0):
            j = i + 1

            while(j < cantDeFilas(A) and paso):
                if(fila_actual[h] < absoluto(A[j][h])):
                    intercambiarFilas(A, i, j)
                    paso = False

                j+= 1
            
            h+= 1

        k = i + 1
        a_copia = A.copy()

        while(k < cantDeFilas(a_copia)):
            m_identidad = matriz_identidad(cantDeFilas(a_copia))
            a = a_copia[k][i]
            b = a_copia[i][i]

            if(b != 0):
                m_identidad[k][i] = -(a/b)

            if(m_identidad[k][i] != 0):
                a_copia = productoM(m_identidad, a_copia)
            k += 1

        i+= 1
        A = a_copia
                
    return A

def matriz_identidad(n):
    res = matrizNulaV2(n, n)
    res = np.array(res, dtype=float)

    i = 0

    while(i < n):
        res[i][i] = 1
        i+= 1

    return res

def absoluto(x):

    if(x < 0):
        x = -x

    return x
    
   
def productoM(A, B):
     
    res = matrizNulaV2(cantDeFilas(A), cantDeCol(B))
    transB = traspuesta(B)
    i = 0
     
    while(i < cantDeFilas(A)):
       
        j = 0
         
        while(j < cantDeFilas(transB)):
            k = 0
            acumulador = 0
             
            while(k < A[i].size):
                acumulador += A[i][k]*transB[j][k]
                k += 1
            
            res[i][j] = acumulador
            
            j+= 1
        
        i += 1
       
    return res

def sumaM(A, B):
    res = matrizNula(A) 
    res = np.array(res)
    i = 0
    
    while(i < cantDeFilas(A)):
        j = 0

        while(j < A[i].size):
            res[i][j] = A[i][j] + B[i][j]
            j+= 1

        i += 1

    return res

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

def matrizNula(arr):
    res = []
    i = 0
    
    while(i < cantDeFilas(arr)):
        j = 0
        nueva_lista = []
        while(j < cantDeCol(arr)):
            nueva_lista.append(0)
            j+= 1

        res.append(nueva_lista)
        i += 1

    res = np.array(res)

    return res
    

def cantDeFilas(arr):
    contador = 0
    for elem in arr:
        contador += 1

    return contador

def cantDeCol(arr):
    return arr[0].size
    

v = np.array([1,2,3])
m = matrizVandermonde(v)

print(sucesionFibo(6))
print(matrizHilbert(2))

