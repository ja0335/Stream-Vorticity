import os;
import os.path;
import sys;
from mpl_toolkits.mplot3d import Axes3D;
import numpy as np;
import matplotlib.pyplot as plt;
from matplotlib import cm;
from matplotlib.ticker import LinearLocator


basepath = os.path.dirname(__file__)
fig = plt.figure()

#for PlotNum in xrange(2):
for PlotNum in xrange(1):
    PlotTittle = ""

    if PlotNum == 0:
        # filepath = os.path.abspath(os.path.join(basepath, "..", "omega.csv"));
        # PlotTittle = "Vorticity Function ";
    #elif PlotNum == 1:
        filepath = "phi.csv"
        PlotTittle = "Stream Function "
        
    f = open(filepath)
    line = f.readline()
    params = line.split(';')
    N = int(params[0].split('=')[1])
    REYNOLDS_NUMBER = params[1].split('=')[1]
    TIME = params[3].split('=')[1].strip()

    PlotTittle = PlotTittle + "\nRe=" + REYNOLDS_NUMBER + "\nTime=" + TIME + " sec"

    xlist = np.linspace(0,1, N)
    ylist = np.linspace(0,1, N)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.zeros([N, N])

    lineNumber = 0
    MaxXCoord = 0
    MaxYCoord = 0
    MimXCoord = 0
    MimYCoord = 0
    MinValue = sys.maxint
    MaxValue = -sys.maxint

    print "Start reading files info..."
    
    while line:
        # jump first line cause it is about file info
        line = f.readline()
        Zstr = line.split(';')
        Length = len(Zstr) - 1
        
        for i in xrange(Length):
            value = float(Zstr[i])
            Z[lineNumber][i] = value

            if value > MaxValue: 
                MaxValue = value
                MaxXCoord = i
                MaxYCoord = lineNumber+1

            if value < MinValue:
                MinValue = value
                MimXCoord = i
                MimYCoord = lineNumber+1
                
        lineNumber += 1
       
    f.close()
    
    print "End reading files info..."
    print "The Max Value is located at Row=" + str(MaxXCoord) + ", Col=" + str(MaxYCoord)
    print "The Min Value is located at Row=" + str(MimXCoord) + ", Col=" + str(MimYCoord)
    
    levels = np.linspace(MinValue, MaxValue, 1000)
    #ax = fig.add_subplot(1, 2, PlotNum+1, projection='3d')
    ax = fig.add_subplot(1, 1, PlotNum + 1, projection='3d')
    plt.title(PlotTittle)

    #makes the plot square
    plt.gca().set_aspect('equal', adjustable='box')
    
    Dx = np.zeros(Z.shape);
    Dy = np.zeros(Z.shape);

    print Dx.shape;
    #for d in xrange(len(Z)):


    #surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False);
    #surf = ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, antialiased=False);

    #colortuple = ('y', 'b')
    #colors = np.empty(X.shape, dtype=str)
    #for y in range(len(Y)):
    #    for x in range(len(X)):
    #        colors[x, y] = colortuple[(x + y) % len(colortuple)]

    #surf = ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0)
    ax.set_zlim3d(MinValue, MaxValue)

#plt.show()