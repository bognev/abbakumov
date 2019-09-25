import numpy as np

A = np.array([[2,-2,3],[6,-7,14],[4,-8,30]])
E21 = np.array([[1,0,0],[-3,1,0],[0,0,1]])
E31 = np.array([[1,0,0],[0,1,0],[-2,0,1]])
E32 = np.array([[1,0,0],[0,1,0],[0,-4,1]])

U = E32@E31@E21@A
print(U)
print((E21^-1)@(E31^-1)@(E32^-1)@U)