import numpy as np

def array_response(a1,a2,N):
    y = np.zeros((N,1),dtype=complex)
    for m in range(int(np.sqrt(N))):
        for n in range(int(np.sqrt(N))):
            y[m*(int(np.sqrt(N)))+n] = np.exp(1j*np.pi*(m*np.sin(a1)*np.sin(a2) + n*np.cos(a2)))
    y = y/np.sqrt(N)
    y = np.ravel(y) # Convent to 1-dimention, MAY BE NOT THE BEST WAY
    return y