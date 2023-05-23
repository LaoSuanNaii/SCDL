from train import train_model
from ssa import SSA
import numpy as np

lb = [3, 1, 3, 2, 2, 2, 2, 2, 2]
ub = [9, 9, 5, 12, 32, 20, 12, 12, 12]
lb = np.array(lb)[None, :]
ub = np.array(ub)[None, :]
fMin, bestX = SSA(lb, ub, lb.shape[-1], train_model)
print('best acc：', fMin)
print('best hyperparameter：', bestX)



