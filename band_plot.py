import numpy as np
import matplotlib.pyplot as plt
Fermi_Energy=0
E=np.loadtxt("./Energy.txt")
kpath=np.loadtxt("./K-PATH.txt")
for i in range(np.shape(E)[0]):
    plt.plot(kpath[:],E[i,:]-Fermi_Energy,color="black")
plt.axhline(0,alpha=0.5,color='black')
plt.xlim(0,1)
plt.ylim(-5,10)
plt.show()
