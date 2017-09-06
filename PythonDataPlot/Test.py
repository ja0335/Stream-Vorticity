import itertools

# of course this is a fake one just to offer an example
def source():
    return itertools.cycle((1,2,3,4,5,6,7,8,9,10))

import pylab as plt
import numpy as np
import scipy as sp

# Generate some test data, i.e. our "observations" of the signal
N = 100
vals = source()
X = np.array([vals.next() for _ in xrange(N)])

# Compute the FFT
W    = np.fft.fft(X)
freq = np.fft.fftfreq(N,1)

# Look for the longest signal that is "loud"
threshold = 10**2
idx = np.where(abs(W)>threshold)[0][-1]

max_f = abs(freq[idx])
print "Period estimate: ", 1/max_f

plt.subplot(311)
plt.scatter([max_f,], [np.abs(W[idx]),], s=100,color='r')
plt.plot(freq[:N/2], abs(W[:N/2]))
plt.xlabel(r"$f$")

plt.subplot(312)
plt.plot(1.0/freq[:N/2], abs(W[:N/2]))
plt.scatter([1/max_f,], [np.abs(W[idx]),], s=100,color='r')
plt.xlabel(r"$1/f$")
plt.xlim(0,20)


plt.subplot(313)
plt.plot(np.array([i for i in xrange(len(X))]), X);

plt.show()