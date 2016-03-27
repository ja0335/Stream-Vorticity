import os
import os.path
import sys;
import time;
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

basepath = os.path.dirname(__file__)

fig = plt.figure();

def animate(i): 
    fig.clear();

    contour1 = 0; 
    contour2 = 0;

    t0 = time.time();
    print "Start writing frame" + str(i);

    for PlotNum in xrange(2):
    
        PlotTittle = "";

        if PlotNum == 0:
            filepath = os.path.abspath(os.path.join(basepath, "..", "x64/bin/Release/Data/phi_" + str(i+1) + ".csv"));
            PlotTittle = "Stream Function ";
        else:
            filepath = os.path.abspath(os.path.join(basepath, "..", "x64/bin/Release/Data/omega_" + str(i+1) + ".csv"));
            PlotTittle = "Vorticity Function ";
        
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
        
            for j in xrange(Length):
                value = float(Zstr[j]);
                Z[lineNumber][j] = value;

                if value > MaxValue: 
                    MaxValue = value;

                if value < MinValue:
                    MinValue = value;
                
            lineNumber += 1;
       
        f.close();
    
        levels = np.linspace(MinValue, MaxValue, 1000);
        plt.subplot(1, 2, PlotNum+1);
        plt.title(PlotTittle);
        plt.gca().set_aspect('equal', adjustable='box')
        
        if PlotNum == 0:
            contour1 = plt.contour(X, Y, Z, levels);
        else:
            contour2 = plt.contour(X, Y, Z, levels);

            
    print "Finished writing frame"+ str(i) + ". Elapsed time: " + str(time.time() - t0);
    
    return contour1, contour2;
	

anim = animation.FuncAnimation(fig, animate, frames=60002);
anim.save('Stream-Vorticity.mp4', fps=30, writer="ffmpeg", codec="libx264");