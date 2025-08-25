from librerias import esCuadrada, triangSup, cantDeCol, triangInf, diagonal, traza, traspuesta
from librerias import esSimetrica, caclcularAx, productoM, cantDeFilas, intercambiarFilas
from librerias import sumar_fila_multiplo, esDiagonalmenteDominante, matrizCirculante, matrizVandermonde
from librerias import numeroAureo, sucesionFibo, matrizFibonacci, matrizHilbert
import numpy as np

def test_es_cuadrada():

    M = np.array([[1,2,3],[3,4,2],[2,4,2]])
    assert esCuadrada(M) == True


def test_triangsup():
    
    M = np.array([[1,2,3],[3,4,2],[2,4,2]])
    triang_sup = np.array([[0,2,3],[0,0,2],[0,0,0]])
    m_res = triangSup(M)
    res_compa = True
    i = 0
    
    while(i < cantDeCol(M)):
        j = 0 
        while(j < cantDeCol(M)):
            if(m_res[i][j] != triang_sup[i][j]):
                res_compa = False
            
            j+= 1

        i += 1
    
    assert res_compa == True

def test_trianginf():
    M = np.array([[1,2,3],[3,4,2],[2,4,2]])
    triang_inf = np.array([[0,0,0],[3,0,0],[2,4,0]])
    m_res = triangInf(M)
    res_compa = True
    i = 0
    
    while(i < cantDeCol(M)):
        j = 0 
        while(j < cantDeCol(M)):
            if(m_res[i][j] != triang_inf[i][j]):
                res_compa = False
            
            j+= 1

        i += 1
    
    assert res_compa == True

def test_diagonal():
    M = np.array([[1,2,3],[3,4,2],[2,4,2]])
    d = np.array([[1,0,0],[0,4,0],[0,0,2]])
    m_res = diagonal(M)
    res_compa = True
    i = 0
    
    while(i < cantDeCol(M)):
        j = 0 
        while(j < cantDeCol(M)):
            if(m_res[i][j] != d[i][j]):
                res_compa = False
            
            j+= 1

        i += 1
    
    assert res_compa == True

def test_traza():
    M = np.array([[1,2,3],[3,4,2],[2,4,2]])
    
    assert traza(M) == 7

def test_transpuesta():
    M = np.array([[1,2,3],[3,4,2],[2,4,2]])
    t = np.array([[1,3,2],[2,4,4],[3,2,2]])
    m_res = traspuesta(M)
    res_compa = True
    i = 0
    
    while(i < cantDeCol(M)):
        j = 0 
        while(j < cantDeCol(M)):
            if(m_res[i][j] != t[i][j]):
                res_compa = False
            
            j+= 1

        i += 1
    
    assert res_compa == True

def test_es_simetrica():
    M = np.array([[-2,1,3],[1,-3,2],[3,2,0]])
     
    assert esSimetrica(M) == True 


def test_calcularAx():
    A = np.array([[5,4],[2,1]])
    x = np.array([[1],[2]])
    b = productoM(A,x)
    b_esperado = np.array([[13],[4]])

    print(b)
    print(b_esperado)

    es_el_res = True
    i = 0

    while(i < cantDeFilas(b) and es_el_res):
        j = 0

        while(j < b[i].size and es_el_res):

            if(b[i][j] != b_esperado[i][j]):
                es_el_res = False

            j += 1

        i+= 1
    
    assert es_el_res == True

def test_intercambiar_filas():

    A = np.array([[5,2],[1,4],[2,52]])
    intercambiarFilas(A,0, 2)
     
    assert A[0][0] == 2
    assert A[0][1] == 52
    assert A[2][0] == 5
    assert A[2][1] == 2


def test_sumar_fila_multiplo():

    A = np.array([[5,2],[1,4],[2,52]])
    sumar_fila_multiplo(A,0, 1, 2)
    
    assert A[0][0] == 7
    assert A[0][1] == 10


def test_es_diagonalmente_dominante():
    
    A = np.array([[5,2],[1,1]])
    assert esDiagonalmenteDominante(A) == False

    B = np.array([[5,2,-2],[1,20,-5],[1, 5, -7]])
    assert esDiagonalmenteDominante(B) == True

def test_matriz_circulante():
    v = np.array([1,2,3])
    m_circulante = matrizCirculante(v)

    assert m_circulante[0][0] == 1
    assert m_circulante[0][1] == 2
    assert m_circulante[0][2] == 3
    assert m_circulante[1][0] == 3
    assert m_circulante[1][1] == 1
    assert m_circulante[1][2] == 2
    assert m_circulante[2][0] == 2
    assert m_circulante[2][1] == 3 
    assert m_circulante[2][2] == 1
    
def test_matriz_vandermonde():
    v = np.array([1,2,3])
    m_vandermonde= matrizVandermonde(v)

    assert m_vandermonde[0][0] == 1
    assert m_vandermonde[0][1] == 1
    assert m_vandermonde[0][2] == 1
    assert m_vandermonde[1][0] == 1
    assert m_vandermonde[1][1] == 2
    assert m_vandermonde[1][2] == 3
    assert m_vandermonde[2][0] == 1
    assert m_vandermonde[2][1] == 4
    assert m_vandermonde[2][2] == 9

def test_numero_aureo():
    a = sucesionFibo(4)
    b = sucesionFibo(5)
    assert numeroAureo(4) == b/a
    
def test_matriz_fibo():
    m_fibo = matrizFibonacci(4)

    assert m_fibo[0][0] == 1 #11 = 2
    assert m_fibo[0][1] == 2 #12 = 3
    assert m_fibo[0][2] == 3 #13 = 4
    assert m_fibo[0][3] == 5 #14 = 5
    assert m_fibo[1][0] == 2 #21 = 3
    assert m_fibo[1][1] == 3 #22 = 4
    assert m_fibo[1][2] == 5 #23 = 5
    assert m_fibo[1][3] == 8 #24 = 6
    assert m_fibo[2][0] == 3 #31 = 4
    assert m_fibo[2][1] == 5 #32 = 5
    assert m_fibo[2][2] == 8 #33 = 6
    assert m_fibo[2][3] == 13 #34 = 7
    assert m_fibo[3][0] == 5 #41 = 5
    assert m_fibo[3][1] == 8 #42 = 6
    assert m_fibo[3][2] == 13 #43 = 7
    assert m_fibo[3][3] == 21 #44 = 8

def test_matriz_hilbert():
    m_hilbert = matrizHilbert(4)

    assert m_hilbert[0][0] == 1/3 #11 = 2
    assert m_hilbert[0][1] == 1/4 #12 = 3
    assert m_hilbert[0][2] == 1/5 #13 = 4
    assert m_hilbert[0][3] == 1/6 #14 = 5
    assert m_hilbert[1][0] == 1/4 #21 = 3
    assert m_hilbert[1][1] == 1/5 #22 = 4
    assert m_hilbert[1][2] == 1/6 #23 = 5
    assert m_hilbert[1][3] == 1/7 #24 = 6
    assert m_hilbert[2][0] == 1/5 #31 = 4
    assert m_hilbert[2][1] == 1/6 #32 = 5
    assert m_hilbert[2][2] == 1/7 #33 = 6
    assert m_hilbert[2][3] == 1/8 #34 = 7
    assert m_hilbert[3][0] == 1/6 #41 = 5
    assert m_hilbert[3][1] == 1/7 #42 = 6
    assert m_hilbert[3][2] == 1/8 #43 = 7
    assert m_hilbert[3][3] == 1/9 #44 = 8

#def test_evaluar_polinomio():
    