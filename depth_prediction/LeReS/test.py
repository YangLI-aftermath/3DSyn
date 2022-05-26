import numpy as np
a = np.random.randn(3,3)
b = np.random.randn(3,3)
print(np.stack([a,b],axis=2))