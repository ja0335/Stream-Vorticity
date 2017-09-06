#ifndef _CUDA_KERNELS_H_
#define _CUDA_KERNELS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>
#include "Macros.h"

void DeviceQuery();

void CopyDataFromHostToDevice(Real * Data_h, Real * Data_d);

void CopyDataFromDeviceToHost(Real * Data_h, Real * Data_d);

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
	Real * u,
	Real * u_d,
	Real * v,
	Real * v_d,
	Real * phi,
	Real * phi_d,
	Real * w,
	Real * w_d);

void SOR(Real * omega_d, Real * phi_d, Real * w_d, Real h, Real Beta, cudaDeviceProp CudaDeviceProp);

Real UpdateVorticity(
	Real * omega_d,
	Real * u_d,
	Real * v_d,
	Real * max_d,
	Real * max_h,
	Real * phi_d,
	Real * w_d,
	Real h,
	Real Viscocity,
	cudaDeviceProp CudaDeviceProp);

void FillPixels(
	sf::Uint8* Pixels_h,
	sf::Uint8* Pixels_d,
	Real * Data_d,
	Real MinValue,
	Real MaxValue,
	cudaDeviceProp CudaDeviceProp);

#endif