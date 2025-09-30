#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
import numpy as np

def calculaLU(A):
    cant_op = 0           #Preguntar que operaciones se consideran parte de la cantidad.
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()

    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    ## desde aqui -- CODIGO A COMPLETAR
    i = 0
    while(i < n):
        elem_diagonal = Ac[i][i]

        if(elem_diagonal == 0):
            return None, None, 0

        j = i + 1
        while(j < m):
            h = i + 1
            escalar = Ac[j][i]/elem_diagonal
            cant_op += 1
            Ac[j][i] = escalar
            
            while(h < n):
                Ac[j][h] = Ac[j][h] - escalar*Ac[i][h]
                h+= 1
                cant_op += 2

            j+= 1
        
        i+=1

    
    I = matriz_identidad(Ac.shape[0])
    L = triangInf(Ac) + I
    U = triangSup(Ac) + diagonal(Ac)
    return L, U, cant_op

    ## hasta aqui, calculando L, U y la cantidad de operaciones sobre 
    ## la matriz Ac

            
def res_tri(L, b, inferior = True):
    xs = []
    
    if(inferior):
        i = 1
        xs.append(np.float64(b[0]/L[0][0]))
        while(i < L.shape[0]):
            j = 0
            h = 0
            xi = b[i]
            while(j < len(xs)):
                xi -= np.float64(L[i][j]*xs[h])
                h += 1
                j+= 1

            xi = np.float64(xi/L[i][j])
            xs.append(xi)
            i += 1
    else:
        i = L.shape[0] - 2
        xs.append(np.float64(b[b.size - 1]/L[L.shape[0] - 1][L.shape[0] - 1]))
        while(i >= 0):
            j = L.shape[0] - 1
            h = 0
            xi = b[i]

            while(h < len(xs)):
                xi-= np.float64(L[i][j]*xs[h])
                h+= 1
                j-= 1
            
            xi = np.float64(xi/L[i][j])
            xs.append(xi)
            i-= 1 

        k = len(xs) - 1
        xs_copia = []
        while(k >= 0):
            xs_copia.append(xs[k])
            k -= 1

        xs = xs_copia

    xs = np.array(xs, dtype = float)
    return xs

def inversa(A):
    L, U, nu = calculaLU(A)
    inversa_a = matrizNulaV2(A.shape[0], A.shape[1])
    det_u = 1
    i = 0

    #Aqui chequeamos si A es inversible
    if(nu == 0):
        return None

    #Desde aqui calculamos la inversa
    canonico_num = 0
    col_actual = 0
    while(canonico_num < A.shape[1]):
        fil_actual = 0
        canonico = crear_canonico(canonico_num, A.shape[1])
        sol_l = res_tri(L, canonico)
        sol_u = res_tri(U, sol_l, inferior = False)
        
        while(fil_actual < inversa_a.shape[1]):
            inversa_a[fil_actual][col_actual] = sol_u[fil_actual]
            fil_actual += 1

        col_actual += 1
        canonico_num += 1

    return inversa_a

def calculaLDV(A):
    L, U, ne = calculaLU(A)
    
    if(ne == 0):
        return None, None, None, 0
    
    U_transpuesta = traspuesta(U)
    V_trans, D, ne = calculaLU(U_transpuesta)
    
    if(ne == 0):
        return None, None, None, 0
    
    V = traspuesta(V_trans)

    return L, D, V, 1
        

def esSDP(A, atol = 1e-10):
    res = esSimetrica(A)

    if(res):
        L, D, V, ne = calculaLDV(A)
        
        if(ne == 0):
            return None
        i = 0

        while(i < D.shape[0] and res):
            if(D[i][i] < atol):
                res = False

            i+= 1

    
    return res


#   res = productoM(productoM(np.linalg.inv(LU[1]), np.linalg.inv(LU[0])), np.array

def main():
    n = 7
    B = np.eye(n) - np.tril(np.ones((n,n)),-1) 
    B[:n,n-1] = 1
    print('Matriz B \n', B)
    
    L,U,cant_oper = elim_gaussiana(B)
    
    print('Matriz L \n', L)
    print('Matriz U \n', U)
    print('Cantidad de operaciones: ', cant_oper)
    print('B=LU? ' , 'Si!' if np.allclose(np.linalg.norm(B - L@U, 1), 0) else 'No!')
    print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )

#if __name__ == "__main__":
#    main()

def triangSup(A):

    if(A.shape[1]== 1 and A.shape[0] == 1):
        return A
    
    actual = 1
    posicion = 0
    U = matrizNula(A)
    #U = np.array(U)

    while(actual < A.shape[1]):
        j = actual
        while(j < A.shape[1]):
            U[posicion][j] = A[posicion][j]
            j+= 1

        actual += 1
        posicion += 1

    return U

def triangInf(A):

    if(A.shape[1] == 1 and A.shape[0] == 1):
        return A
    
    actual = 0
    posicion = 1
    L = matrizNula(A)
    #res = np.array(res)

    while(actual <= A.shape[1]//2):
        j = 0
        while(j <= actual):
            elem = np.float64(A[posicion][j])
            L[posicion][j] = elem
            j+= 1

        actual += 1
        posicion += 1

    return L


def diagonal(A):
    i = 0
    D = matrizNula(A)
    #res = np.array(res)

    while(i < A.shape[1]):
        D[i][i] = A[i][i]
        i += 1

    return D

def matrizNula(arr):

    res = []
    i = 0
    
    while(i < arr.shape[0]):
        j = 0
        nueva_lista = []
        while(j < arr.shape[1]):
            nueva_lista.append(0)
            j+= 1

        res.append(nueva_lista)
        i += 1

    res = np.array(res, dtype = float)

    return res

def matriz_identidad(n):
    res = matrizNulaV2(n, n)
    #res = np.array(res, dtype=float)

    i = 0

    while(i < n):
        res[i][i] = 1
        i+= 1

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

def esSimetrica(A):
    res = False

    if(A.shape[0] == A.shape[1]):
        trans = traspuesta(A)
        res = True
        i = 0

        while(i < A.shape[1] and res):
            j = 0
            while(j < A[i].size and res):
                if(A[i][j] != trans[i][j]):
                    res = False
                j += 1

            i += 1

    return res

A = np.array([[1,2,3],[4,5,6],[7,8,9]])


print(inversa(A))

