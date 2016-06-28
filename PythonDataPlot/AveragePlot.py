import os;
import os.path;
import sys;
from mpl_toolkits.mplot3d import Axes3D;
import numpy as np;
import matplotlib.pyplot as plt;



basepath = os.path.dirname(__file__)
fig = plt.figure()

filepath = os.path.abspath(os.path.join(basepath, "..", "x64/Release/Data/average.csv"));

f = open(filepath);
line = f.readline();
data = line.split(';');

plt.xlabel(data[0]);
plt.ylabel(data[1]);

X = [];
Y = [];

while line:
    line = f.readline();
    data = line.split(';');

    if(len(data) == 2):
        X.append(data[0]);
        Y.append(data[1]);
        
       
f.close();
#plt.plot(X, Y, 'o');
plt.plot(X, Y);
plt.show();
