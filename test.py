import matplotlib.pyplot as plt
import numpy as np
 
x = np.linspace(-3,3)
y = x**2

fig = plt.figure(tight_layout=True)
 
axes = fig.subplots(2, 3)
 
axes[1,1].plot(x, y)
 
plt.show()