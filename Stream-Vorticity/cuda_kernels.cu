#include <iostream>
#include "cuda_kernels.h"

#include "Functions.h"

void DeviceQuery()
{
	printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess)
	{
		printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
	{
		printf("There are no available device(s) that support CUDA\n");
	}
	else
	{
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}

	int dev, driverVersion = 0, runtimeVersion = 0;

	for (dev = 0; dev < deviceCount; ++dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

		// Console log
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

		printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);


#if CUDART_VERSION >= 5000
		// This is supported in CUDA 5.0 (runtime API device properties)
		printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
		printf("  Memory Bus Width:                              %d-bit\n", deviceProp.memoryBusWidth);

		if (deviceProp.l2CacheSize)
		{
			printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
		}

#else
		// This only available in CUDA 4.0-4.2 (but these were only exposed in the CUDA Driver API)
		int memoryClock;
		getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
		printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
		int memBusWidth;
		getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
		printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
		int L2CacheSize;
		getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

		if (L2CacheSize)
		{
			printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
		}

#endif

		printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
			deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
			deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
		printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
			deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
		printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
			deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);


		printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
		printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
		printf("  Warp size:                                     %d\n", deviceProp.warpSize);
		printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
		printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
		printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxGridSize[0],
			deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
		printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
		printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
		printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
		printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
		printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n", deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
		printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
		printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

		const char *sComputeMode[] =
		{
			"Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
			"Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
			"Prohibited (no host thread can use ::cudaSetDevice() with this device)",
			"Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
			"Unknown",
			NULL
		};
		printf("  Compute Mode:\n");
		printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
	}
}

// -------------------------------------------------------------------------
void CopyDataFromHostToDevice(Real * Data_h, Real * Data_d)
{
	sf::Uint64 size = GRID_SIZE * GRID_SIZE * sizeof(Real);

	cudaMemcpy(Data_d, Data_h, size, cudaMemcpyHostToDevice);
}

void CopyDataFromDeviceToHost(Real * Data_h, Real * Data_d)
{
	sf::Uint64 size = GRID_SIZE * GRID_SIZE * sizeof(Real);

	cudaMemcpy(Data_d, Data_h, size, cudaMemcpyDeviceToHost);
}

void CopyDataFromHostToDevice(
	Real * omega,
	Real * omega_d,
	Real * phi,
	Real * phi_d,
	Real * w,
	Real * w_d)
{
	sf::Uint64 size = GRID_SIZE * GRID_SIZE * sizeof(Real);

	cudaMemcpy(omega_d, omega, size, cudaMemcpyHostToDevice);
	cudaMemcpy(phi_d, phi, size, cudaMemcpyHostToDevice);
	cudaMemcpy(w_d, w, size, cudaMemcpyHostToDevice);
}

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
	Real * w_d)
{
	sf::Uint64 size = GRID_SIZE * GRID_SIZE * sizeof(Real);

	cudaDeviceSynchronize();

	cudaMemcpy(omega, omega_d, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(phi, phi_d, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(u, u_d, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(v, v_d, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(w, w_d, size, cudaMemcpyDeviceToHost);
}

// -------------------------------------------------------------------------
__global__ void SOR_Even_kernel(Real * omega, Real * phi, Real * w, Real h, Real Beta)
{
	int idx = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);

	if (idx >= GRID_SIZE * GRID_SIZE)
		return;

	w[idx] = phi[idx];

	if (idx % 2 == 0 &&
		idx >= GRID_SIZE && idx < GRID_SIZE * GRID_SIZE - GRID_SIZE
		&&
		idx % GRID_SIZE != 0 && (idx + 1) % GRID_SIZE != 0)
	{
		phi[idx] = 0.25f*Beta*(phi[idx + 1] + phi[idx - 1] + phi[idx + GRID_SIZE] + phi[idx - GRID_SIZE] + h*h*omega[idx]) + (1.0f - Beta)*phi[idx];
	}
}

__global__ void SOR_Odd_kernel(Real * omega, Real * phi, Real * w, Real h, Real Beta)
{
	int idx = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);

	if (idx >= GRID_SIZE * GRID_SIZE)
		return;

	w[idx] = phi[idx];

	if (idx % 2 != 0 &&
		idx >= GRID_SIZE && idx < GRID_SIZE * GRID_SIZE - GRID_SIZE
		&&
		idx % GRID_SIZE != 0 && (idx + 1) % GRID_SIZE != 0)
	{
		phi[idx] = 0.25f*Beta*(phi[idx + 1] + phi[idx - 1] + phi[idx + GRID_SIZE] + phi[idx - GRID_SIZE] + h*h*omega[idx]) + (1.0f - Beta)*phi[idx];
	}
}

void SOR(Real * omega_d, Real * phi_d, Real * w_d, Real h, Real Beta, cudaDeviceProp CudaDeviceProp)
{
	int NumBlocks = static_cast<int>(ceil((GRID_SIZE * GRID_SIZE) / static_cast<Real>(CudaDeviceProp.maxThreadsPerBlock)));
	//We need at least 1 block
	NumBlocks = (NumBlocks == 0) ? 1 : NumBlocks;

	dim3 ThreadsPerBlock(
		static_cast<unsigned int>(sqrt(CudaDeviceProp.maxThreadsPerBlock)), 
		static_cast<unsigned int>(sqrt(CudaDeviceProp.maxThreadsPerBlock)));

	SOR_Even_kernel << <NumBlocks, ThreadsPerBlock >> >(omega_d, phi_d, w_d, h, Beta);
	cudaDeviceSynchronize();
	SOR_Odd_kernel << <NumBlocks, ThreadsPerBlock >> >(omega_d, phi_d, w_d, h, Beta);
	cudaDeviceSynchronize();
}

// -------------------------------------------------------------------------
__global__ void UpdateVorticity_kernel1(Real * omega, Real * u, Real * v, Real * sum, Real * phi, Real * w, Real h, Real Viscocity)
{
	int idx = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);

	if (idx >= GRID_SIZE * GRID_SIZE)
		return;

	if (idx < GRID_SIZE)
	{
		// -------------------------------------------------------------------------
		// boundary conditions for the Vorticity
		omega[IJ(idx, GRID_SIZE - 1)] = -2.0f*phi[IJ(idx, GRID_SIZE - 2)] / (h*h) - LID_SPEED * (2.0f / h); // top wall
		omega[IJ(idx, 0)] = -2.0f*phi[IJ(idx, 1)] / (h*h); // bottom wall
		omega[IJ(GRID_SIZE - 1, idx)] = -2.0f*phi[IJ(GRID_SIZE - 2, idx)] / (h*h); // left wall
		omega[IJ(0, idx)] = -2.0f*phi[IJ(1, idx)] / (h*h); // right wall
	}

	__syncthreads();

	if (idx >= GRID_SIZE && idx < GRID_SIZE * GRID_SIZE - GRID_SIZE
		&&
		idx % GRID_SIZE != 0 && (idx + 1) % GRID_SIZE != 0)
	{
		// --------------------------------------------------------------------------
		// RHS Calculation
		u[idx] =  (phi[idx + GRID_SIZE] - phi[idx - GRID_SIZE]) / (2 * h);
		v[idx] = -(phi[idx + 1] - phi[idx - 1]) / (2 * h);
		
		w[idx] = - ( u[idx] * ((omega[idx + 1] - omega[idx - 1]) / (2 * h))
				   + v[idx] * ((omega[idx + GRID_SIZE] - omega[idx - GRID_SIZE]) / (2 * h)))
			+ Viscocity * (omega[idx + 1] + omega[idx - 1] + omega[idx + GRID_SIZE] + omega[idx - GRID_SIZE] - 4.0f*omega[idx]) / (h*h);
		
		sum[idx] = u[idx] + v[idx];
	}
}

// -------------------------------------------------------------------------
__global__ void UpdateVorticity_kernel2(Real * omega, Real * w, Real dt)
{
	int idx = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);

	if (idx >= GRID_SIZE * GRID_SIZE)
		return;
	
	if (idx >= GRID_SIZE && idx < GRID_SIZE * GRID_SIZE - GRID_SIZE
		&&
		idx % GRID_SIZE != 0 && (idx + 1) % GRID_SIZE != 0)
	{
		// -------------------------------------------------------------------------
		// Update the vorticity
		omega[idx] = omega[idx] + dt * w[idx];
	}
}

// -------------------------------------------------------------------------
__global__ void ReduceMax_kernel(Real * data)
{
	extern __shared__ float sdata[];

	int idx = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);
	int tid = threadIdx.x + blockDim.x*(threadIdx.y);

	if (idx < GRID_SIZE * GRID_SIZE)
		sdata[tid] = data[idx];
	else
		sdata[tid] = -9999999;

	__syncthreads();

	// in-place reduction in global memory
	for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1)
	{
		if (tid < s && sdata[tid] < sdata[tid + s])
		{
			//sdata[tid] = sdata[tid + s];
		}
		__syncthreads();        // make sure all adds at one stage are done!
	}

	// only thread 0 writes result for this block back to global mem
	if (tid == 0)
	{
		data[blockIdx.x] = 31415;// sdata[0];
	}
}


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
	cudaDeviceProp CudaDeviceProp)
{
	int NumBlocks = static_cast<unsigned int>(ceil((GRID_SIZE * GRID_SIZE) / static_cast<Real>(CudaDeviceProp.maxThreadsPerBlock)));
	//We need at least 1 block
	NumBlocks = (NumBlocks == 0) ? 1 : NumBlocks;

	dim3 ThreadsPerBlock(
		static_cast<unsigned int>(sqrt(CudaDeviceProp.maxThreadsPerBlock)), 
		static_cast<unsigned int>(sqrt(CudaDeviceProp.maxThreadsPerBlock)));

	UpdateVorticity_kernel1 << <NumBlocks, ThreadsPerBlock >> >(omega_d, u_d, v_d, max_d, phi_d, w_d, h, Viscocity);
	cudaDeviceSynchronize();
	ReduceMax_kernel << <NumBlocks, ThreadsPerBlock, CudaDeviceProp.maxThreadsPerBlock * sizeof(Real) >> >(max_d);
	cudaDeviceSynchronize();
	
	CopyDataFromDeviceToHost(max_h, max_d);
	WriteArray(max_h, CudaDeviceProp.maxThreadsPerBlock, "Data/max.txt");
	Real gpu_max = -9999999;

	for (int i = 0; i < NumBlocks; i++)
	{
		Real temp = max_h[i];
		if (gpu_max < temp)
			gpu_max = temp;
	}

	// Get an apropiate dt
	Real dt = (8 * REYNOLDS_NUMBER * h * h) / (16 + gpu_max * gpu_max * REYNOLDS_NUMBER * REYNOLDS_NUMBER * h * h);

	UpdateVorticity_kernel2 << <NumBlocks, ThreadsPerBlock >> >(omega_d, w_d, DT);
	cudaDeviceSynchronize();

	return dt;
}

// -------------------------------------------------------------------------
__global__ void TransportDye_kernel(const sf::Uint8* DyeBackBuffer, sf::Uint8* DyeFrontBuffer, const Real * u)
{
	int idx = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);

	if (idx >= GRID_SIZE * GRID_SIZE)
		return;

	if (idx >= GRID_SIZE && idx < GRID_SIZE * GRID_SIZE - GRID_SIZE
		&&
		idx % GRID_SIZE != 0 && (idx + 1) % GRID_SIZE != 0)
	{
		//sf::Uint8 X = (idx % GRID_SIZE) + 1;
		DyeFrontBuffer[idx * 4 + 0] = DyeBackBuffer[idx * 8 + 0];
		DyeFrontBuffer[idx * 4 + 1] = DyeBackBuffer[idx * 8 + 1];
		DyeFrontBuffer[idx * 4 + 2] = DyeBackBuffer[idx * 8 + 2];
		DyeFrontBuffer[idx * 4 + 3] = DyeBackBuffer[idx * 8 + 3];
	}
}

__global__ void SwapDyeBuffers_kernel(sf::Uint8* DyeBackBuffer, const sf::Uint8* DyeFrontBuffer)
{
	int idx = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);

	if (idx >= GRID_SIZE * GRID_SIZE)
		return;

	if (idx >= GRID_SIZE && idx < GRID_SIZE * GRID_SIZE - GRID_SIZE
		&&
		idx % GRID_SIZE != 0 && (idx + 1) % GRID_SIZE != 0)
	{
		//sf::Uint8 X = (idx % GRID_SIZE) + 1;
		DyeBackBuffer[idx * 4 + 0] = DyeFrontBuffer[idx * 4 + 0];
		DyeBackBuffer[idx * 4 + 1] = DyeFrontBuffer[idx * 4 + 1];
		DyeBackBuffer[idx * 4 + 2] = DyeFrontBuffer[idx * 4 + 2];
		DyeBackBuffer[idx * 4 + 3] = DyeFrontBuffer[idx * 4 + 3];
	}
}

// -------------------------------------------------------------------------
__global__ void FillPixels_kernel(sf::Uint8* Pixels, const Real * InData, Real MinValue, Real MaxValue)
{
	int idx = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);

	if (idx >= GRID_SIZE * GRID_SIZE)
		return;

	if (InData[idx] < 0.0f)
	{
		Pixels[idx * 4 + 0] = 0;
		Pixels[idx * 4 + 1] = 0;
		Pixels[idx * 4 + 2] = 255;
		Pixels[idx * 4 + 3] = 255;
	}
	else if (InData[idx] == 0.0f)
	{
		Pixels[idx * 4 + 0] = 255;
		Pixels[idx * 4 + 1] = 255;
		Pixels[idx * 4 + 2] = 255;
		Pixels[idx * 4 + 3] = 255;
	}
	else if (InData[idx] > 0.0f)
	{
		Pixels[idx * 4 + 0] = 255;
		Pixels[idx * 4 + 1] = 0;
		Pixels[idx * 4 + 2] = 0;
		Pixels[idx * 4 + 3] = 255;
	}
}

void FillPixels(
	sf::Uint8* Pixels_h,
	sf::Uint8* Pixels_d,
	Real * Data_d,
	Real MinValue,
	Real MaxValue,
	cudaDeviceProp CudaDeviceProp)
{
	int NumBlocks = static_cast<unsigned int>(ceil((GRID_SIZE * GRID_SIZE) / static_cast<Real>(CudaDeviceProp.maxThreadsPerBlock)));
	//We need at least 1 block
	NumBlocks = (NumBlocks == 0) ? 1 : NumBlocks;

	dim3 ThreadsPerBlock(
		static_cast<unsigned int>(sqrt(CudaDeviceProp.maxThreadsPerBlock)), 
		static_cast<unsigned int>(sqrt(CudaDeviceProp.maxThreadsPerBlock)));

	FillPixels_kernel << <NumBlocks, ThreadsPerBlock >> >(Pixels_d, Data_d, MinValue, MaxValue);

	cudaMemcpy(Pixels_h, Pixels_d, GRID_SIZE * GRID_SIZE * 4 * sizeof(sf::Uint8), cudaMemcpyDeviceToHost);
}
