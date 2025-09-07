import matplotlib.pyplot as plt                             #np.meshgrid: Se usa para crear una cuadricula.         
import numpy as np                                          #np.linespace(ini. del invervalo, fin del interalo, cant. de puntos en el intervalo): Se usa para crear puntos equispaciados.
import pandas as pd                                         #w1.reshape(n, m): Crear una matriz de n filas y columnas a partir de un vector
                                                            #np.concatenate(A, B, axis = 1 o axis = 0) : 1 para concatenar por col. y 0 por filas.
def pointsGrid(esquinas):
    # crear 10 lineas horizontales
    [w1, z1] = np.meshgrid(np.linspace(esquinas[0,0], esquinas[1,0], 46),
                        np.linspace(esquinas[0,1], esquinas[1,1], 10))

    [w2, z2] = np.meshgrid(np.linspace(esquinas[0,0], esquinas[1,0], 10),
                        np.linspace(esquinas[0,1], esquinas[1,1], 46))

    w = np.concatenate((w1.reshape(1,-1),w2.reshape(1,-1)),1)
    z = np.concatenate((z1.reshape(1,-1),z2.reshape(1,-1)),1)
    wz = np.concatenate((w,z))
                         
    return wz

def proyectarPts(T, wz):
    assert(T.shape == (2,2)) # chequeo de matriz 2x2
    assert(T.shape[1] == wz.shape[0]) # multiplicacion matricial valida   
    xy = wz
    xy = T@xy                                                   #T@xy :estamos transformando xy para generar el subesapcio imagen. El resultado tiene como columnas a los 
                                                                #generadores de la imagen
    return xy

          
def vistform(T, wz, titulo=''):
    # transformar los puntos de entrada usando T
    xy = proyectarPts(T, wz)
    if xy is None:
        print('No fue implementada correctamente la proyeccion de coordenadas')
        return
    # calcular los limites para ambos plots
    minlim = np.min(np.concatenate((wz, xy), 1), axis=1)
    maxlim = np.max(np.concatenate((wz, xy), 1), axis=1)

    bump = [np.max(((maxlim[0] - minlim[0]) * 0.05, 0.1)),
            np.max(((maxlim[1] - minlim[1]) * 0.05, 0.1))]
    limits = [[minlim[0]-bump[0], maxlim[0]+bump[0]],
               [minlim[1]-bump[1], maxlim[1]+bump[1]]]             

    fig, (ax1, ax2) = plt.subplots(1, 2)         
    fig.suptitle(titulo)
    grid_plot(ax1, wz, limits, 'w', 'z')    
    grid_plot(ax2, xy, limits, 'x', 'y')   
    plt.show() 


def grid_plot(ax, ab, limits, a_label, b_label):
    ax.plot(ab[0,:], ab[1,:], '.')
    ax.set(aspect='equal',
           xlim=limits[0], ylim=limits[1],
           xlabel=a_label, ylabel=b_label)


def main():
    print('Ejecutar el programa')
    # generar el tipo de transformacion dando valores a la matriz T
    T = np.array([[1, 3],[3, 1]])
    #T = pd.read_csv('T.csv', header=None).values      #T es la matriz de las transformacion 
    corners = np.array([[0,0],[100,100]])                               
    # corners = np.array([[-100,-100],[100,100]]) array con valores positivos y negativos
    wz = pointsGrid(corners)
    vistform(T, wz, 'Deformar coordenadas')
    
    
#if __name__ == "__main__":
#    main()

"""
Parte del eje. 1)
Para graficar el efecto de f(x,y) = (x/2,y/2) cambien T de main() por np.array([[1/2, 0], [0, 1/2]])
Para graficar el efecto de la inversa cambien T por np.array([[2, 0],[0, 2]])

"""


"""
Ejercicio 2)
    1) res: T = [[a, 0], [0, b]] y T^-1 = [[1/a, 0],[0, 1/b]]

    2) res: Si a = 2 y b = 3 --> T = [[2, 0], [0, 3]] y T^-1 = [[1/2, 0],[0, 1/3]]

        T: 
        Si lo aplicamos a los vectores canonico obtenemos, T*[[1],[0]] = [[2, 0], [0, 3]]*[[1],[0]] = [[2],[0]]  
                                                           T*[[0],[1]] = [[2, 0], [0, 3]]*[[0],[1]] = [[0],[3]]

        Si lo aplicamos al punto arbitrario (1, 1) obtenemos T*[[1],[1]] = [[2, 0], [0, 3]]*[[1],[1]] = [[2], [3]]

        Si lo aplicamos a una circunferencia tenemos que hacerlo en dos partes y obtenemos: T*[[sqrt(y**2 - 1)], [y]] = [[2*sqrt(y**2 - 1)], 3*y]
                                                                               T*[[-sqrt(y**2 - 1)], [y]] = [[-2*sqrt(y**2 - 1)], 3*y]


        T^-1:
        Si lo aplicamos a los vectores canonico obtenemos, T*[[1],[0]] = [[1/2, 0], [0, 1/3]]*[[1],[0]] = [[1/2],[0]]  
                                                           T*[[0],[1]] = [[1/2, 0], [0, 1/3]]*[[0],[1]] = [[0],[1/3]]
        
        Si lo aplicamos al punto arbitrario (1, 1) obtenemos T*[[1],[1]] = [[1/2, 0], [0, 1/3]]*[[1],[1]] = [[1/2], [1/3]]

        Si lo aplicamos a una circunferencia tenemos que hacerlo en dos partes y obtenemos: T*[[sqrt(y**2 - 1)], [y]] = [[(1/2)*sqrt(y**2 - 1)], (1/3)*y]
                                                                               T*[[-sqrt(y**2 - 1)], [y]] = [[(-1/2)*sqrt(y**2 - 1)], (1/2)*y]

        
        Aun no lo grafique, pero creo que se puede usar geogebra para ver cuales son los puntos que tiene asignalos circunferencia en el espacio de llegada.
       
"""

"""
Ejercicio 3)
        1) res: T = [[1, c], [d, 1]] y T-1 = [[1/(1 - dc), -c/(1 - dc)], [-d/(1 - dc), 1/(1 - dc)]]
        

"""

"""
Ejercicio 4)
        1) res: Las funciones son a(tita) = cos(tita), b(tita) = cos(pi/2 + tita), c(tita) = sen(tita) y d(tita) = sen(pi/2 + tita)
            R(tita) = [[cos(tita), sen(pi/2 + tita)], [sen(tita), cos(pi/2 + tita)]]

        2) La diferencia que noto es que las rotaciones mantienen la forma de los cuadraditos, mientras que la deformacion de cizalla los estira y los deforma.

"""



#Ejercicios del modulo:
def rota(theta):
    M = np.array([[np.cos(theta), np.cos(np.pi/2 + theta)], [np.sin(theta), np.sin(np.pi/2 + theta)]])
    return M

def escala(s):
    i = 0
    l = []
    
    while(i < len(s)):
        l.append([])
        i+= 1

    j = 0
    while(j < len(s)):
        k = 0

        while(k < len(s)):
            l[j].append(0)
            k+= 1

        j+= 1

    h = 0

    while(h < len(s)):
        l[h][h] = s[h]
        h += 1

    return np.array(l)


def rota_y_escala(theta, s):
    M = rota(theta)
    A = escala(s)
    return A@M


def afin(theta, s, b):
    M = rota_y_escala(theta, s)
    res = np.array([[0,0,0],[0,0,0],[0,0,1]])

    res[0][0] = M[0][0]
    res[0][1] = M[0][1]
    res[1][0] = M[1][0]
    res[1][1] = M[1][1]
    res[0][2] = b[0]
    res[1][2] = b[1]

    return res

def trans_afin(v, theta, s, b):
    B = rota_y_escala(theta, s)
    v_columna = np.array([[v[0]],[v[1]]])
    res_col = B@v_columna
    res_v = np.array([res_col[0][0], res_col[1][0]])
    res_v[0] += b[0]
    res_v[1] += b[1]
    return res_v
    




#print(afin(angulo, s, b))
#print(trans_afin(v, angulo, s, b))
#print(afin(np.pi/2,np.array([2,2]), np.array([3,3])))
