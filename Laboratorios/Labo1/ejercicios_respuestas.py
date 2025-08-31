import sys
import numpy as np

"""
Ejercicio 1.1 y 1.2)
¿En ambos casos se obtiene el resultado esperado? ¿Porque?
    res: Si. Esto es porque sabemos que se sigue la convencion de IEEE 754 
    para representar numeros reales (Los numeros abstractos) en maquina
    ,porque se que python redondea (Usa el redondeo) los numeros que no puede representar en maquina
    ,y porque los numeros en punto flotante en python (El python de la version de mi pc) tienen precision doble.
    Por lo tanto podemos operar como la haria la computadora para conseguir el resultado usando la info. de arriba.


  
    Escriban el numero 0.25 en base 2. ¿Como queda expresado en terminos
    de su mantisa y exponente?


    res: Primero, tenemos que separa 0.25 en su parte entera y parte decimal.   <--- Esto es para mis compas;eros de grupo
        0 es la parte entera y 0.25 es la parte decimal.

        Ahora tenemos que buscar la representacion en base 2 de 0:

            - Vamos a divir 0 por 2 hasta conseguir su resto y su cociente 
            ,y despues divideremos el cociente obtenido por 2 hasta conseguir 
            su cociente y resto, y asi sucesivamente hasta que ya no se pueda.

            - A partir de los restos obtenidos de la diviones obtendremos
            la representacion en base 2 del 0 que es (0)_10 = (0)_2   <--- 0 en base 10 es igual a 0 en base 2

            - Eje:
                11 | 11/2 ---> resto = 1 y cociente = 5
                5 | 5/2 ---> resto = 1 y cociente = 2
                2 | 2/2 ---> resto = 0 y cociente = 0
                1| 1/2 ---> resto = 1 y cociente = 0  <---- Ya no se puede seguir dividiendo

                Agarramos los restos y obtenemos (11)_10 = (1011)_2
            
            
        Ahora buscaremos la representacion en base 2 de 0.25:

            - Vamos a multiplicar 0.25 por 2. Una vez que tenemos el resultado lo 
              vamos a descomponer en una suma de un entero mas un valor con decimales.
              Despues agarraremos el valor decimal de la descomposicion y repetiremos
              el proceso hasta que en un punto la parte decimal sera 0.

            - Eje:
                0.25 | 0.25*2 = 0.50 | 0.50 = 0 + 0.50    

                0.50 | 0.50*2 = 1 | 1 = 1 + 0   <---- La parte decimal es cero

                Agarramos las partes enteras y obtenemos (0.25)_10 = (10)_2


            Por lo tanto (0.25)_10 = (0.01)_2

            
            {} Ahora queremos expresarlo en terminos de mantisa y exponente:

            Pasos para representarlo en ese termino:
                1. Expresar el numero en base 10 en base 2 (Ya lo hicimos)
                2. Escribirlo en notacion cientifica en base 2
 
            
            Sabemos que 0.25 en base 10 es 0.01 en base 2.
            0.01 en notacion cientifica es 1.001*2^1  <-- Si multiplicamos por 2^1, desplazamos cada bit
                                                          de 001 un lugar a izquierda y nos quedara 0.01
                                                          El uno de la izquierda del punto siempre va en notacion cientifica en base 2
            
            Obutvimos 1.001*2^1 y esta expresado en terminos de matiza y exponente. La matiza es 001 y el exponente es 1
"""

"""
Ejercicio 1.3)
     Escriban el numero 0.3 en base 2. ¿Que dificultades aparecen al escribir
     0.3 en binario? ¿Se puede escribir exactamente con una mantisa finita?

     res: Noto que despues de llegar a cierto punto usando el metodo multiplicar por 2 para representar a 0.3 en base 2, se vuelven a repetir los valores 
     que descompuse anteriormente y nunca llego a una descomposicion en donde la parte decimal es 0. Al parecer tiene una representacion infinita
     

     Podria ser. No estoy seguro.

"""

"""
Ejercicio 2)
¿Cuanto da (√2)^2 − 2?
    res: Si lo calculo a mano me daria 0

¿Cuanto da en python?
Dio 4.440892098500626e-16

Prueba de que raiz_cuadrada(2x**2 + 1) - 1 es igual a 2x**2/(raiz_cuadrada(2x**2 + 1) - 1) y vicerversa
c.a:
2x**2/(raiz_cuadrada(2x**2 + 1) + 1) = 2x**2 * (raiz_cuadrada(2x**2 + 1) - 1)/(raiz_cuadrada(2x**2 + 1) + 1)*(raiz_cuadrada(2x**2 + 1) - 1)
= 2x**2 * (raiz_cuadrada(2x**2 + 1) - 1)/(raiz_cuadrada(2x**2 + 1)**2 - 1**2)
= 2x**2 * (raiz_cuadrada(2x**2 + 1) - 1)/(2x**2 + 1 - 1)
= 2x**2 * (raiz_cuadrada(2x**2 + 1) - 1)/2x**2
= raiz_cuadrada(2x**2 + 1) - 1 
"""




