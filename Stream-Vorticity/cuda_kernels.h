#ifndef _CUDA_KERNELS_H_
#define _CUDA_KERNELS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>
#include "Macros.h"

void DeviceQuery();

void CopyDataFromHostToDevice(
	Real * omega,
	Real * omega_d,
	Real * phi,
	Real * phi_d,
	Real * w,
	Real * w_d);

void CopyDataFromDeviceToHost(
	Real * omega,
	Real * omega_d,
	Real * phi,
	Real * phi_d,
	Real * w,
	Real * w_d);

void SOR(Real * omega_d, Real * phi_d, Real * w_d, Real h, Real Beta, cudaDeviceProp CudaDeviceProp);

void UpdateVorticity(
	Real * omega_d,
	Real * phi_d,
	Real * w_d,
	Real h,
	Real Viscocity,
	cudaDeviceProp CudaDeviceProp);

#endif