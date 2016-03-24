#pragma once

#define N 9
#define PLOT_RESOLUTION 1024
#define IJ(i, j) ((i) + N*(j))

//#define SINGLE_PRECISION

#ifdef SINGLE_PRECISION
#define Number float
#else
#define Number double
#endif