
"""
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
"""
import numpy as np
figsize = (16, 9)


## Dejamos las funciones sin implementar en este archivo para que coloquen las suyas.
# Para que funcione con el resto de este codigo tienen que hacer una pequeña modificacion en householder para guardar las Q y R intermedias

def gram_schmidt(A, tol=1e-5):
    Q = []
    R = matrizNulaV2(A.shape[0], A.shape[1])
    q_1 = A[0: A.shape[0],0]
    R[0][0] = norma_2(q_1)
    q_1 = q_1/norma_2(q_1)
    Q.append(q_1)
    j = 1

    while(j < A.shape[0]):
        col_a_actual = A[0: A.shape[0], j]
        k = 0

        while(k < j):
            r_kj = producto_interno(Q[k], col_a_actual)
            R[k][j] = r_kj
            col_a_actual = col_a_actual - r_kj*Q[k]
            k = k + 1
        
        rjj = norma_2(col_a_actual)
        R[j][j] = rjj 
        col_a_actual = col_a_actual*(1/norma_2(col_a_actual))
        Q.append(col_a_actual)
        j = j + 1

    Q = np.array(Q, dtype= float)
    Q = traspuesta(Q)
    print(Q@R)
    
    return Q, R

def householder_qr(A, tol=1e-5,extras=False):
    R = A.copy()
    Q = matriz_identidad(A.shape[0])
    extra_info = {"R_matrices":[R],"Q_matrices":[Q]}
    k = 0
    
    while(k < A.shape[1]):
        x = R[k: A.shape[0], k]
        alfa = (-signo(x[0]))*norma_2(x)
        u = x - alfa*crear_canonico(0, x.size)
        
        if(norma_2(u) > tol):
            u = u/norma_2(u)
            m_householder = matriz_de_reflexion(u, A.shape[0], k)
            if(k > 0):
                m_householder = matriz_h(m_householder, A.shape[0], k)

            extra_info["R_matrices"].append(productoM(m_householder, R))
            extra_info["Q_matrices"].append(productoM(Q, traspuesta(m_householder)))
            R = productoM(m_householder, R)
            Q = productoM(Q, traspuesta(m_householder))
            
        k+= 1
    
    if(extras):
        return Q, R, extra_info
    else:
        return Q, R

    """
    QR con reflectores de Householder sobre las columnas de la matriz A.
    extras : bool, opcional
        Si es True, devuelve informacion extra sobre el proceso de factorizacion.
        Por defecto es False. Esto lo hacemos para poder graficar el proceso.
    Devuelve la factorización QR de A usando reflectores de Householder.
    Devuelve: 
        Q, R, extra_info (si extras es True)
        Q, R (si extras es False)
    extra_info es un diccionario con la clave:s:
        "R_matrices": lista de las matrices R en cada paso
        "Q_matrices": lista de las matrices Q en cada paso
    """
    None

def matriz_h(A, m, k):
    I = matriz_identidad(m)
    submatriz_izq = I[k: m, 0: k]
    submatriz_up = I[0:k, 0: m]

    H = np.concatenate((submatriz_izq, A), axis = 1)
    H = np.concatenate((submatriz_up, H), axis=0)

    return H
    

def matriz_de_reflexion(u, m, k):
    I = matriz_identidad(m-k)
    u = np.array([u], dtype = float)
    u_t = traspuesta(u)
    u_m = (-2)*productoM(u_t, u)
    Hk_moño = I + u_m

    return Hk_moño

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

def matriz_identidad(n):
    res = matrizNulaV2(n, n)
    #res = np.array(res, dtype=float)

    i = 0

    while(i < n):
        res[i][i] = 1
        i+= 1

    return res

def signo(a):
    res = None

    if(a < 0):
        res = -1
    else:
        res = 1
    return res

def norma_2(v):
    res = 0
    i = 0

    while(i < v.size): 
        res = res + (v[i])**2 
        i += 1; 
        
    return np.sqrt(res)

def producto_interno(v, w):
    res = 0
    i = 0

    while(i < v.size):
        res = res + v[i]*w[i]
        i = i + 1

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
A = np.array([[1,1,1],[1, 2, 3],[0,0,1]])


"""
def graficar_descomposicion(A):
    Q,R = gram_schmidt(A)
    print("Close" ,np.allclose(A, Q@R))
    plt.figure(figsize=figsize)
    plt.imshow(Q)
    plt.title("Q")
    plt.show()
    plt.figure(figsize=figsize)
    plt.imshow(Q.T @ Q)
    plt.title("Q.T @ Q")
    plt.show()
    plt.figure(figsize=figsize)
    plt.imshow(R)
    plt.title("R")
    plt.show()


def show_subespace(vs, ax=None, color="blue", alpha=0.4):
    v1 = vs[0]/np.linalg.norm(vs[0])
    if len(vs) ==1:
        v2 = v1*0
    else:
        v2 = vs[1]/np.linalg.norm(vs[1])

    a = np.linspace(-2, 2, 30)
    b = np.linspace(-2, 2, 30)
    A, B = np.meshgrid(a, b)
    X = A * v1[0] + B * v2[0]
    Y = A * v1[1] + B * v2[1]
    Z = A * v1[2] + B * v2[2]
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    if len(vs)==1:
        ax.plot(X,Y,Z)
    else:
        ax.plot([X[0,0], X[0,-1]], [Y[0,0], Y[0,-1]], [Z[0,0], Z[0,-1]], color='k', linewidth=1)
        ax.plot([X[0,0], X[-1,0]], [Y[0,0], Y[-1,0]], [Z[0,0], Z[-1,0]], color='k', linewidth=1)
        ax.plot([X[-1,0], X[-1,-1]], [Y[-1,0], Y[-1,-1]], [Z[-1,0], Z[-1,-1]], color='k', linewidth=1)
        ax.plot([X[0,-1], X[-1,-1]], [Y[0,-1], Y[-1,-1]], [Z[0,-1], Z[-1,-1]], color='k', linewidth=1)
        ax.plot_surface(X, Y, Z, alpha=alpha, linewidth=1, rstride=1, cstride=1, color=color)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    lims = 1.2
    ax.set_xlim(-lims, lims)
    ax.set_ylim(-lims, lims)
    ax.set_zlim(-lims, lims)
    return ax


# toma una lista de vectores entonces si queremos graficar los vectores columna de una matriz A hacemos plot_vectors(A.T)
def plot_vectors(vectors, ax=None, color="black"):
    origin = np.zeros(3)
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    for vec in vectors:
        ax.quiver(*origin, *vec, length=1.0, normalize=False, color=color, alpha=0.7, linewidths=3.5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    return ax



if __name__ == "__main__":
    U = np.array([[-1,1,1],[1,-1,0.3], [0.5,0.4,1]]) # esta es la matriz que vamos a usar, pueden probar con otras


    show_gramm_schmidt = True
    if show_gramm_schmidt:

        ax = plot_vectors(U.T)
        plt.title(f"Vectores columna de la matriz")
        plt.show()

        Q,R = gram_schmidt(U)
        for i in range(U.shape[1]):
            ax = None
            vs = U.T[:i+1]
            ax = plot_vectors(vs)
            ax = plot_vectors(Q.T[:i+1], ax, color="red")
            if i > 0:
                ax = show_subespace(U.T[:i],ax, alpha=0.1)
            plt.title(f"gram_shmidt: Ortogonalizando la componente {i}")
            plt.show()



    U = U[:,:2]  # sacamos uno de los vectores para ser mas facil de entender pero podriamos dejar los 3
    ax = plot_vectors(U.T)
    plt.title(f"Vectores columna de la matriz")
    plt.show()

    Q,R, extra_info = householder_qr(U, extras=True)


    Q_matrices = extra_info["Q_matrices"]

    for i in range(U.shape[1]):
        ax = None
        vs = U.T
        qs = Q_matrices[i].T
        ax = plot_vectors(vs)
        ax = plot_vectors(qs, ax, color="red")
        ax = show_subespace(qs, ax, color="red", alpha=0.1)

        if i == 0:
            ax.plot([qs[i][0], vs[i][0]], [qs[i][1], vs[i][1]], [qs[i][2], vs[i][2]], color="orange", linestyle="dashed")
        if i == 1:
            # eliminamos la componente de vs[1] en la direccion de qs[0]
            v_proj = np.dot(vs[1], qs[0]) * qs[0]
            v_ortho = vs[1] - v_proj
            ax.plot([0, v_ortho[0]], [0, v_ortho[1]], [0, v_ortho[2]], color="green", linestyle="dashed", label="Componente ortogonal")
            ax.plot([v_ortho[0], qs[1][0]], [v_ortho[1], qs[1][1]], [v_ortho[2], qs[1][2]], color="orange", linestyle="dashed", label="Componente paralelo")
        plt.title(f"householder: Ortogonalizando la componente {i}")
        plt.show()

        


    qs = Q.T[:2] # Nuestro algoritmo en realidad obtiene una base de R3 pero descartamos la ultima columna de Q y de R para la representacion reducida (habra 0s en esa direccion)
    ax = plot_vectors(vs)
    ax = plot_vectors(qs, ax, color="red")
    ax = show_subespace(qs, ax, color="red", alpha=0.1)
    plt.title(f"householder: Resultado final")
    plt.show()
"""


