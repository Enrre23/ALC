import matplotlib.pyplot as plt
import numpy as np
from librerias import generar_puntos_equispaciados, evaluar_polinomio, numeroAureo
a = [1,2,3]
b = [1, 2, 3]

#n: Indica el polonomio que se va a evaluar (n = 1, n = 2 o n = 3).
#m: Es la cantidad de puntos que se va a graficar
def graficar_puntos_poli(n, m):

#Ejercicio16)
    p = generar_puntos_equispaciados(m)
    y = []


    if(n == 0):

        i = 0
        while(i < len(p)):
            y.append(evaluar_polinomio(n, p[i]))
            i+= 1

        plt.plot(p, y)
        plt.show()

    if(n == 1):

        i = 0
        while(i < len(p)):
            y.append(evaluar_polinomio(n, p[i]))
            i+= 1

        plt.plot(p, y)
        plt.show()
        
    
    if(n == 2):

        i = 0
        while(i < len(p)):
            y.append(evaluar_polinomio(n, p[i]))
            i+= 1

        plt.plot(p, y)
        plt.show()

#Ejercicio14)
def graficar_numero_oro(n):
    i = 1
    h = []
    x = []
    while(i < n):
        x.append(i)
        h.append(numeroAureo(i))
        i+= 1

    plt.plot(x, h)
    plt.show()



graficar_numero_oro(20)