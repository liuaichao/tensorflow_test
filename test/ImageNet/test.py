#-*-coding:utf-8-*-
import numpy as np
a = np.array([1,2,3])
b = np.array([4,5,6])

np.save('./test.npy', a)
np.load('./test.npy', 'r+')

np.save('./test.npy', b)
print(np.load('./test.npy'))