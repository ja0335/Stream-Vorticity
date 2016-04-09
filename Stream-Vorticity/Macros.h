#ifndef _MACROS_H_
#define _MACROS_H_

#define PI 3.14159265358979323846f

// dx = 0.1 => 11
// dx = 0.05 => 21
// dx = 0.01 => 100
// dx = 0.005 = 201
#define GRID_SIZE 201 
#define LID_SPEED 1.0f
#define REYNOLDS_NUMBER 10000.0f
#define MAX_SOR_ITERATIONS 100
#define SOR_TOLERANCE_ERROR 0.001f
#define DT 0.005f // time step

#define CAPTURE_DATA 0
#define SINGLE_PRECISION 0
#define USE_CPP_PLOT 0
#define PLOT_RESOLUTION 1024
#define USE_CUDA 1

#define IJ(i, j) ((i) + GRID_SIZE*(j))

#if SINGLE_PRECISION
#define Real float
#else
#define Real double
#endif

#endif