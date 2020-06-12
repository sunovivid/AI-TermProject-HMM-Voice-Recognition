import numpy as np

m = np.array([-100.0,-5.0,-np.inf])
tp = np.array([ [-3.0,-np.inf,-6.0],
                [-5.0,-2.0,-64.0],
                [-np.inf,-1.0,-0.0] ])
print(m + tp)
print(np.max(m + tp, axis=0))
print(np.shape(np.max(m + tp, axis=0)))
print(np.transpose(np.max(m + tp, axis=0)))
print(np.shape(np.transpose(np.max(m + tp, axis=0))))