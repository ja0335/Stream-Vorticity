#ifndef _MACROS_H_
#define _MACROS_H_

#define PI 3.14159265358979323846f

// dx = 0.1 => 11
// dx = 0.05 => 21
// dx = 0.01 => 100
// dx = 0.005 = 201
#define GRID_SIZE 513 
#define LID_SPEED 1.0f
#define REYNOLDS_NUMBER 15000//9376.0f
#define MAX_SOR_ITERATIONS 100
#define SOR_TOLERANCE_ERROR 0.001f
//#define DT 0.001f // time step

#define CAPTURE_DATA 0
#define PRINT_DATA 0
#define SINGLE_PRECISION 0
#define USE_CPP_PLOT 1
#define PLOT_RESOLUTION GRID_SIZE
#define USE_CUDA 1
#define USE_DYE 0
#define NUM_PARTICLES 600
#define IJ(i, j) ((i) + GRID_SIZE*(j))

#if SINGLE_PRECISION
#define Real float
#else
#define Real double
#endif

#endif