import os;
import os.path;
import sys;
from mpl_toolkits.mplot3d import Axes3D;
import numpy as np;
import matplotlib.pyplot as plt;



basepath = os.path.dirname(__file__)

Build = "Release"; #"Release";

filepath1 = os.path.abspath(os.path.join(basepath, "..", "x64/" + Build + "/Data/u.csv"));
filepath2 = os.path.abspath(os.path.join(basepath, "..", "x64/" + Build + "/Data/v.csv"));
        
f1 = open(filepath1);
f2 = open(filepath2);

line1 = f1.readline();
line2 = f2.readline();

params = line1.split(';');
N = int(params[0].split('=')[1]);
REYNOLDS_NUMBER = params[1].split('=')[1];
TIME = params[3].split('=')[1].strip();


xlist = np.linspace(0,1, N)
ylist = np.linspace(0,1, N)
X, Y = np.meshgrid(xlist, ylist)
U = np.zeros([N, N]);
V = np.zeros([N, N]);

lineNumber = 0;

while line1:
    # jump first line1 cause it is about file info
    line1 = f1.readline();
    line2 = f2.readline();
    
    Ustr = line1.split(';');
    Vstr = line2.split(';');
    Length = len(Ustr)-1;
        
    for i in xrange(Length):
        U[lineNumber][i] = float(Ustr[i]);
        V[lineNumber][i] = float(Vstr[i]);
                
    lineNumber += 1;
       
f1.close();
f2.close();
    
plt.figure()
#plt.title(PlotTittle);
PlotEach = N/20;
    
plt.gca().set_aspect('equal', adjustable='box')

plt.quiver(X[::PlotEach, ::PlotEach], Y[::PlotEach, ::PlotEach], U[::PlotEach, ::PlotEach], V[::PlotEach, ::PlotEach],
               edgecolor='k', facecolor='None', linewidth=.5)
plt.show()