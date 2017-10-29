#ifndef _MACROS_H_
#define _MACROS_H_

#define PI 3.14159265358979323846f

#define GRID_SIZE 301 
#define LID_SPEED 1.0f
#define REYNOLDS_NUMBER 12500
#define USE_CPP_PLOT 0
#define MAX_SOR_ITERATIONS 100
#define SOR_TOLERANCE_ERROR 0.00001f
#define CONVERGENCE_ERROR 0.000001f
#define SUCCESIVE_STEPS_FOR_CONVERGENCE 100 // how much steps of convergence error to assume problem has converged


#define CAPTURE_DATA 0
#define PRINT_DATA 0
#define SINGLE_PRECISION 0
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