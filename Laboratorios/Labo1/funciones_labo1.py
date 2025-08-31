import numpy as np
import matplotlib.pyplot as plt
from funciones_labo0 import esSimetrica

#Ejercicio 2)
def generar_puntos_equispaciados(n):
    i = 0  
    res = []  
    
    while(i < n):                               
        res.append((10**14) + ((10**16 - 10**14)/(n - 1))*i)  
        i += 1              

    return res


def func_ejer_2(): 
    x = generar_puntos_equispaciados(100)
    p1 = []
    p2 = []

    i = 0
    j = 0

    while(i < len(x)):
        p1.append(np.sqrt(2*((x[i])**2) + 1) - 1)
        i += 1

    while(j < len(x)):
        p2.append(2*(x[j]**2)/(np.sqrt(2*(x[j]**2) + 1) + 1))
        j += 1

    plt.plot(x, p2)
    plt.show()
    plt.plot(x, p1)
    plt.show()


#Ejercicio 3)
#El limite de la sucesion es raiz de 2


def graficar_suce():
    i = 1
    l = []
    l.append(np.sqrt(2))
    acumulador = np.sqrt(2)

    while(i < 100):
        acumulador = (acumulador*acumulador)/np.sqrt(2) #A partir de la iteracion 46 (i = 46) se comienza a desastabilizar
        l.append(acumulador)                            #En vez de que el resultado de acumulador sea algo aproximado a raiz de 2
        i+= 1                                           #Se esta guardando valores cada mas grandes a raiz de 2. Mucho mas grandes

                                                        #En la iteracion 65 (i = 65) el valor de acumulador sera inf

    plt.plot(l)
    plt.show()



#Ejercicio 4)
"""
Para n = 6 con precesion de 32 bits:
    La primera suma dio: 14.357358
    La segunda suma dio: 15.403683

Para n = 7 con precision de 32 bits:
    La primera suma dio: 15.403683
    La segunda suma dio: 15.403683

Para n = 6 con precesion de 64 bits:
    La primera suma dio: 14.392726722864989
    La segunda suma dio: 16.002164235298594

Para n = 7 con precision de 64 bits:
    La primera suma dio: 16.695311365857272
    La segunda suma dio: 18.304749238293297
"""
#Euler:
def aprox_euler():
    suma = np.float64(0)
    
    for i in range(0, 11):
        suma += np.float64(1/factorial(i))

    print(suma)

    num_euler = np.e
    print(num_euler)


def factorial(n):

    if(n == 1):
        return 1
    
    if(n == 0):
        return 1
    
    return n*factorial(n - 1)
    
#Ejercicio5
def error_lu():
    A = np.array([[4, 2, 1], [2, 7 ,9], [0, 5, 22/3]])
    L = np.array([[1,0,0], [0.5, 1, 0], [0, 5/6, 1]])
    U = np.array([[4,2,1], [0, 6, 8.5], [0, 0, 0.25]])

    print(matricesIguales(A, L@U))



#Ejercicio6
#La matriz T me la invente yo porque no decia nada acerca de T
def es_simetrica_error():
    A = np.array(np.random.rand(4,4)) 
    T = np.array([[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]])

    print(esSimetrica(A.T@A))
    print(esSimetrica(A.T@(A*0.25/0.25)))
    print(esSimetrica(A.T@(A*0.2/0.2)))



#Funciones del modulo:
def error(x, x_a):
    return np.abs(x - x_a)

def error_relativo(x, x_a):
    return np.abs(x - x_a)/np.abs(x)

def matricesIguales(A, B):
    res = True
    print(np.abs(A[0][0] - B[0][0])/np.abs(A[0][0]))
    A_copia = A.astype(float)
    B_copia = B.astype(float)

    if((A_copia.shape)[0] != (B_copia.shape)[0] or (A_copia.shape)[1] != (B_copia.shape)[1]):
        res = False


    if(res):
        i = 0
        epsilon = np.finfo(A_copia[0][0]).eps
        
        while(i < (A_copia.shape)[0] and res):
            j = 0 

            while(j < (A_copia.shape)[1] and res):

                if(B_copia[i][j] != 0):                                                     #Si B_copia[i][j] es 0, usaremos la 
                    n = np.abs(A_copia[i][j] - B_copia[i][j])/np.abs(B_copia[i][j])         #formula de error relativo

                    if(n > epsilon):
                        res = False

                    
                else:
                    if((np.abs(A_copia[i][j] - B_copia[i][j]) > epsilon)):  #Si B_copia[i][j] es 0, usaremos la 
                        res = False                                                             #formula de error abosoluto

                j+= 1
                
            i+= 1

    return res







