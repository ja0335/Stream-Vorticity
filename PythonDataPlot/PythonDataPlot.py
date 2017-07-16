import os;
import os.path;
import sys;
from mpl_toolkits.mplot3d import Axes3D;
import numpy as np;
import matplotlib.pyplot as plt;



basepath = os.path.dirname(__file__)
fig = plt.figure()

Build = "Release"; #"Release";

for PlotNum in xrange(4):
    
    PlotTittle = "";

    if PlotNum == 0:
        filepath = os.path.abspath(os.path.join(basepath, "..", "x64/" + Build + "/Data/phi.csv"));
        PlotTittle = "Stream Function ";
    elif PlotNum == 1:
        filepath = os.path.abspath(os.path.join(basepath, "..", "x64/" + Build + "/Data/omega.csv"));
        PlotTittle = "Vorticity Function ";
    elif PlotNum == 2:
        filepath = os.path.abspath(os.path.join(basepath, "..", "x64/" + Build + "/Data/u.csv"));
        PlotTittle = "u Component ";
    elif PlotNum == 3:
        filepath = os.path.abspath(os.path.join(basepath, "..", "x64/" + Build + "/Data/v.csv"));
        PlotTittle = "v Component ";
        
    f = open(filepath);
    line = f.readline();
    params = line.split(';');
    N = int(params[0].split('=')[1]);
    REYNOLDS_NUMBER = params[1].split('=')[1];
    TIME = params[3].split('=')[1].strip();

    PlotTittle = PlotTittle + "\nRe=" + REYNOLDS_NUMBER + "\nTime=" + TIME + " sec";

    xlist = np.linspace(0,1, N)
    ylist = np.linspace(0,1, N)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.zeros([N, N]);

    lineNumber = 0;
    MinValue = sys.maxint;
    MaxValue = -sys.maxint;

    while line:
        # jump first line cause it is about file info
        line = f.readline();
        Zstr = line.split(';');
        Length = len(Zstr)-1;
        
        for i in xrange(Length):
            value = float(Zstr[i]);
            Z[lineNumber][i] = value;

            if value > MaxValue: 
                MaxValue = value;

            if value < MinValue:
                MinValue = value;
                
        lineNumber += 1;
       
    f.close();
    
    levels = np.linspace(MinValue, MaxValue, 1000, endpoint=False);

    plt.subplot(2, 2, PlotNum+1);
    plt.title(PlotTittle);
    plt.gca().set_aspect('equal', adjustable='box')
    plt.contour(X, Y, Z, levels, linewidths=(0.1, 0.2, 0.3, 0.4, 0.5));
    #plt.plot(spacing, spacing, spacing,levels);


plt.show()