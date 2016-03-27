#pragma once

#define PI 3.14159265358979323846f

// 1/0.1 = 10; 1/0.05 = 20; 1/0.01 = 100; 1/0.005 = 200
#define N 201 
#define LID_SPEED 1.0f
#define REYNOLDS_NUMBER 10000.0f
#define MAX_SOR_ITERATIONS 100
#define SOR_TOLERANCE_ERROR 0.001f
#define DT 0.001f // time step

#define SINGLE_PRECISION 0
#define USE_CPP_PLOT 0
#define CAPTURE_DATA 1

#define PLOT_RESOLUTION 1024

#define IJ(i, j) ((i) + N*(j))

#if SINGLE_PRECISION
#define Real float
#else
#define Real double
#endif