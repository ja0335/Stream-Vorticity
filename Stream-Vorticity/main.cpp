#include <iostream>
#include <sstream>
#include "termcolor.h"
#include <stdio.h>
#include <SFML/Graphics.hpp>
#include <mgl2/mgl.h>
#include "Functions.h"
#ifdef USE_CUDA
#include "cuda_kernels.h"
#endif
using namespace sf;

int main(int argc, char **argv)
{
#if USE_CPP_PLOT
	RenderWindow window(VideoMode(1024, 768), "Navier Stokes");
	RenderTexture Canvas;
	Texture DynamicTexture;
	Sprite SpriteDynamicTexture;

	if (!Canvas.create(2048, 2048))
		return EXIT_FAILURE;
	if (!DynamicTexture.create(PLOT_RESOLUTION, PLOT_RESOLUTION))
		return EXIT_FAILURE;

	SpriteDynamicTexture.setTexture(DynamicTexture);
	DynamicTexture.setSmooth(false);
	float SpriteScale = static_cast<float>(Canvas.getSize().x) / static_cast<float>(PLOT_RESOLUTION);
	SpriteDynamicTexture.scale(SpriteScale, SpriteScale);
	Uint64 PixelsBufferSize = PLOT_RESOLUTION * PLOT_RESOLUTION * 4;
	mglGraph			gr(0, PLOT_RESOLUTION, PLOT_RESOLUTION);
	Uint8* Pixels = new Uint8[PixelsBufferSize];	memset(Pixels, 0, PixelsBufferSize * sizeof(Uint8));
#else
	RenderWindow window(VideoMode(480, 320), "Navier Stokes");
#endif
	
	// --------------------------------------------------------------------------------
	// Set initial parameters
	Real Viscocity = 1.0f / REYNOLDS_NUMBER;

	// relaxation factors
	// SOR converges fastest for a square lattice if Beta = 2 / (1.0f + PI / L)
	// where  is the number of lattice points in the x or y directions
	Real Beta = 2.0f / (1.0f + (PI / GRID_SIZE));

	// Spatial step
	Real h = 1.0f / (Real)(GRID_SIZE - 1);

	// --------------------------------------------------------------------------------
	// Data Buffers
	Uint32 SizeOfData = sizeof(Real)* GRID_SIZE * GRID_SIZE;
	Real * phi = new Real[GRID_SIZE * GRID_SIZE];				memset(phi, 0, SizeOfData);
	Real * omega = new Real[GRID_SIZE * GRID_SIZE];				memset(omega, 0, SizeOfData);
	Real * w = new Real[GRID_SIZE * GRID_SIZE];					memset(w, 0, SizeOfData);

#if USE_CUDA
	// --------------------------------------------------------------------------------
	// CUDA Device Data Buffers
	DeviceQuery();

	int CudaDevice = 0;
	cudaSetDevice(CudaDevice);
	cudaDeviceProp CudaDeviceProp;
	cudaGetDeviceProperties(&CudaDeviceProp, CudaDevice);

	Real * phi_d;
	Real * omega_d;
	Real * w_d;

	cudaMalloc((void **)&phi_d, SizeOfData);	cudaMemset(phi_d, 0, SizeOfData);
	cudaMalloc((void **)&omega_d, SizeOfData);	cudaMemset(omega_d, 0, SizeOfData);
	cudaMalloc((void **)&w_d, SizeOfData);		cudaMemset(w_d, 0, SizeOfData);
#endif

	// run the program as long as the window is open
	bool bSimulateNextFrame = false;
	bool bUseKeyToSimulate = true;
	Uint64 CurrentStep = 0;
	Real SimulationTime = 0.0f;
	Clock Clock1;
	Clock Clock2;

	while (window.isOpen())
	{
		bSimulateNextFrame = false;
		// check all the window's events that were triggered since the last iteration of the loop

		Event event;
		while (window.pollEvent(event))
		{
			// "close requested" event: we close the window
			if (Keyboard::isKeyPressed(Keyboard::Escape) || event.type == Event::Closed)
				window.close();

			if (Keyboard::isKeyPressed(Keyboard::Right))
				bSimulateNextFrame = true;
			if (Keyboard::isKeyPressed(Keyboard::Up))
				bUseKeyToSimulate = !bUseKeyToSimulate;
			if (Keyboard::isKeyPressed(Keyboard::Down))
			{
				std::cout << "*********Writing Data*********" << std::endl;
#if USE_CUDA
				CopyDataFromDeviceToHost(omega, omega_d, phi, phi_d, w, w_d);
#endif
				std::stringstream ss_phi;
				//ss_phi << "phi_" << CurrentStep << ".csv";
				//WriteArray(GRID_SIZE, phi, ss_phi.str().c_str());
				WriteArray(GRID_SIZE, phi, SimulationTime, "Data/phi.csv");

				std::stringstream ss_omega;
				//ss_omega << "omega_" << CurrentStep << ".csv";
				//WriteArray(GRID_SIZE, omega, ss_omega.str().c_str());
				WriteArray(GRID_SIZE, omega, SimulationTime, "Data/omega.csv");
				std::cout << "****Finished Writing Data*****" << std::endl;
			}
		}

		if (bUseKeyToSimulate && !bSimulateNextFrame)
			continue;

#if 1 // Stream-Vorticity Calculation

#if USE_CUDA
		// -------------------------------------------------------------------------
		// streamfunction calculation by SOR iteration
		for (int it = 0; it < 1; it++)
		{
			SOR(omega_d, phi_d, w_d, h, Beta, CudaDeviceProp);
		}

		UpdateVorticity(omega_d, phi_d, w_d, h, Viscocity, CudaDeviceProp);

#else
		// -------------------------------------------------------------------------
		// streamfunction calculation by SOR iteration
		for (int it = 0; it < MAX_SOR_ITERATIONS; it++)
		{
			Real Err = 0.0f;

			for (int i = 0; i < GRID_SIZE; i++)
			{
				for (int j = 0; j < GRID_SIZE; j++)
				{
					w[IJ(i, j)] = phi[IJ(i, j)];

					if (i > 0 && i < GRID_SIZE - 1 && j > 0 && j < GRID_SIZE - 1)
						phi[IJ(i, j)] = 0.25f*Beta*(phi[IJ(i + 1, j)] + phi[IJ(i - 1, j)] + phi[IJ(i, j + 1)] + phi[IJ(i, j - 1)] + h*h*omega[IJ(i, j)]) + (1.0f - Beta)*phi[IJ(i, j)];

					Err += abs(w[IJ(i, j)] - phi[IJ(i, j)]);
				}
			}

			// stop if iteration has converged
			if (Err <= SOR_TOLERANCE_ERROR)
				break;
		}

		// -------------------------------------------------------------------------
		// boundary conditions for the Vorticity
		for (int i = 0; i < GRID_SIZE; i++)
		{
			omega[IJ(i, GRID_SIZE - 1)] = -2.0f*phi[IJ(i, GRID_SIZE - 2)] / (h*h) - LID_SPEED * (2.0f / h); // top wall
			omega[IJ(i, 0)] = -2.0f*phi[IJ(i, 1)] / (h*h); // bottom wall
			omega[IJ(GRID_SIZE - 1, i)] = -2.0f*phi[IJ(GRID_SIZE - 2, i)] / (h*h); // left wall
			omega[IJ(0, i)] = -2.0f*phi[IJ(1, i)] / (h*h); // right wall
		}

		// --------------------------------------------------------------------------
		// RHS Calculation
		for (int i = 1; i < GRID_SIZE - 1; i++)
		{
			for (int j = 1; j < GRID_SIZE - 1; j++)
			{
				w[IJ(i, j)] = -0.25f*((phi[IJ(i, j + 1)] - phi[IJ(i, j - 1)])*(omega[IJ(i + 1, j)] - omega[IJ(i - 1, j)])
					- (phi[IJ(i + 1, j)] - phi[IJ(i - 1, j)])*(omega[IJ(i, j + 1)] - omega[IJ(i, j - 1)])) / (h*h)
					+ Viscocity * (omega[IJ(i + 1, j)] + omega[IJ(i - 1, j)] + omega[IJ(i, j + 1)] + omega[IJ(i, j - 1)] - 4.0f*omega[IJ(i, j)]) / (h*h);
			}
		}
		// -------------------------------------------------------------------------
		// Update the vorticity
		for (int i = 1; i < GRID_SIZE - 1; i++)
		{
			for (int j = 1; j < GRID_SIZE - 1; j++)
				omega[IJ(i, j)] = omega[IJ(i, j)] + DT*w[IJ(i, j)];
		}
#endif

#if CAPTURE_DATA
		// -------------------------------------------------------------------------
		// Capture data
#if USE_CUDA
		CopyDataFromDeviceToHost(omega, omega_d, phi, phi_d, w, w_d);
#endif
		std::stringstream ss_phi;
		ss_phi << "Data/phi_" << CurrentStep << ".csv";
		WriteArray(GRID_SIZE, phi, SimulationTime, ss_phi.str().c_str());

		std::stringstream ss_omega;
		ss_omega << "Data/omega_" << CurrentStep << ".csv";
		WriteArray(GRID_SIZE, omega, SimulationTime, ss_omega.str().c_str());
		// -------------------------------------------------------------------------
#endif

		// increment the time
		SimulationTime += DT;
		CurrentStep++;
#endif
		if (CurrentStep % 2000 == 0)
		{
			std::cout
				<< "Current step: " << CurrentStep << "\t"
				<< "Step time: " << Clock1.restart().asSeconds() << " sec\t"
				<< termcolor::red 
				<< "Simulation time: " << SimulationTime << " sec\t"
				<< termcolor::reset
				<< "Elapsed time: " << Clock2.getElapsedTime().asSeconds() << " sec\n" << std::endl;
		}

#if USE_CPP_PLOT
		Canvas.clear(Color::Black);
#if USE_CUDA
		CopyDataFromDeviceToHost(omega, omega_d, phi, phi_d, w, w_d);
#endif
		Plot2(&gr, Pixels, PixelsBufferSize, phi, omega);
		DynamicTexture.update(Pixels);
		Canvas.draw(SpriteDynamicTexture);
		Canvas.display();

		// --------------------------------------------------------
		// Draw the final image result
		window.clear(Color::Black);

		Vector2f WindowOffset = Vector2f(0.95f, 0.95f);
		Vector2f WindowSize = Vector2f(window.getSize().x * WindowOffset.x, window.getSize().y * WindowOffset.y);
		Vector2f ImageSize = Vector2f(static_cast<float>(Canvas.getSize().x), static_cast<float>(Canvas.getSize().y));

		Sprite FinalSprite;
		FinalSprite.setTexture(Canvas.getTexture());
		float FinalSpriteScale = (WindowSize.y) / ImageSize.y;
		FinalSprite.scale(FinalSpriteScale, FinalSpriteScale);

		Vector2f position;
		position.x = (WindowSize.x * 0.5f) / WindowOffset.x;
		position.x -= (ImageSize.x * FinalSprite.getScale().x) * 0.5f;

		position.y = (WindowSize.y * 0.5f) / WindowOffset.y;
		position.y -= (ImageSize.y * FinalSprite.getScale().y) * 0.5f;

		FinalSprite.move(position);
		window.draw(FinalSprite);

		// end the current frame
		window.display();
#endif
	}

#if USE_CUDA
	cudaFree(omega_d);
	cudaFree(phi_d);
	cudaFree(w_d);
#endif
#if USE_CPP_PLOT
	delete[] Pixels;
#endif
	delete[] omega;
	delete[] phi;
	delete[] w;

	return 0;
}