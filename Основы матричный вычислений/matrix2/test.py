import numpy as np

v  = np.array([ [1, 5, 2], [1, 3, 4]])
v_s = np.argsort(v, axis=1)[:,::-1]

print(v_s)