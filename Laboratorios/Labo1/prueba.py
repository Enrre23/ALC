import numpy as np

n = 7
s = np.float64(0)
for i in range(1, 10**n + 1):
    s = s + np.float64(1/i)

print("suma = ", s)
 
s = np.float64(0)
for i in range(1, 5*10**n + 1):
    s = s + np.float64(1/i)
 
print("suma = ", s)





