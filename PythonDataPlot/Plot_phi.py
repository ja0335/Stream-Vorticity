import os;
import os.path;
import sys;
from mpl_toolkits.mplot3d import Axes3D;
import numpy as np;
import matplotlib.pyplot as plt;
from matplotlib import cm;
from matplotlib.ticker import LinearLocator

class Agent:
    Grid = [];
    i = 0;
    j = 0;
    bMoved = False;

    def __init__(self, InGrid, In_i, In_j):
        self.Grid = InGrid;
        self.i = In_i;
        self.j = In_j;
        self.bMoved = False;

    def Move(self):
        #check neighborhood
        CurrentValue = self.Grid[self.i][self.j];
        TopLeft = self.Grid[self.i-1][self.j-1];
        Top = self.Grid[self.i][self.j-1];
        TopRight = self.Grid[self.i+1][self.j-1];
        Left = self.Grid[self.i-1][self.j];
        Right = self.Grid[self.i+1][self.j];
        BottomLeft = self.Grid[self.i-1][self.j+1];
        Bottom = self.Grid[self.i][self.j+1];
        BottomRight = self.Grid[self.i+1][self.j+1];

        i = self.i
        j = self.j
        
        if CurrentValue < TopLeft:
            CurrentValue = TopLeft;
            i = self.i-1;
            j = self.j-1;
            self.bMoved = True;
        if CurrentValue < Top:
            CurrentValue = Top;
            i = self.i;
            j = self.j-1;
            self.bMoved = True;
        if CurrentValue < TopRight:
            CurrentValue = TopRight;
            i = self.i+1;
            j = self.j-1;
            self.bMoved = True;
        if CurrentValue < Left:
            CurrentValue = Left;
            i = self.i-1;
            j = self.j;
            self.bMoved = True;
        if CurrentValue < Right:
            CurrentValue = Right;
            i = self.i+1;
            j = self.j;
            self.bMoved = True;
        if CurrentValue < BottomLeft:
            CurrentValue = BottomLeft;
            i = self.i-1;
            j = self.j+1;
            self.bMoved = True;
        if CurrentValue < Bottom:
            CurrentValue = Bottom;
            i = self.i;
            j = self.j+1;
            self.bMoved = True;
        if CurrentValue < BottomRight:
            CurrentValue = BottomRight;
            i = self.i+1;
            j = self.j+1;
            self.bMoved = True;

basepath = os.path.dirname(__file__)
fig = plt.figure()

Build = "Release"; #"Release";
filepath = os.path.abspath(os.path.join(basepath, "..", "x64/" + Build + "/Data/phi.csv"));
        
f = open(filepath)
line = f.readline()
params = line.split(';')
N = int(params[0].split('=')[1])
REYNOLDS_NUMBER = params[1].split('=')[1]
TIME = params[3].split('=')[1].strip()

PlotTittle = "\nRe=" + REYNOLDS_NUMBER + "\nTime=" + TIME + " sec"

xlist = np.linspace(0,1, N)
ylist = np.linspace(0,1, N)
X, Y = np.meshgrid(xlist, ylist)
Z = np.zeros([N, N])

lineNumber = 0
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

        if value < MinValue:
            MinValue = value
                
    lineNumber += 1
       
f.close()
print "End reading files info..."

levels = np.linspace(MinValue, MaxValue, 1000, endpoint=False);
ax = fig.add_subplot(1, 2, 1)
plt.title(PlotTittle)
plt.gca().set_aspect('equal', adjustable='box')
plt.contour(X, Y, Z, levels, linewidths=(0.1, 0.2, 0.3, 0.4, 0.5));


ax = fig.add_subplot(1, 2, 2)
Agents = [];

Z = np.abs(np.flip(Z, 0));
for i in xrange(1, Z.shape[0]-1):
    for j in xrange(1, Z.shape[1]-1):
        Agents.append(Agent(Z, i, j));
            
TotalAgents = len(Agents);

print "Moving Agents..."

for i in xrange(N):
    for j in xrange(TotalAgents):
        Agents[j].Move();

            
Vortex = np.zeros(Z.shape);
NumAgents = len(Agents)

print "Writting Vortex array..."

for i in xrange(NumAgents):
    if Agents[i].bMoved == True:
        Vortex[ Agents[i].i ][ Agents[i].j ] = 1;

np.savetxt('Vortex.csv', Vortex, delimiter=';');

plt.imshow(Vortex);

plt.show()