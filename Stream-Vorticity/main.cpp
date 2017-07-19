#include <iostream>
#include <sstream>
#include "termcolor.h"
#include <stdio.h>
#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>
#include "Functions.h"
#include "Particle.h"
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
	Image DyeImage;

	if (!Canvas.create(1024, 1024))
		return EXIT_FAILURE;
	if (!DynamicTexture.create(GRID_SIZE, GRID_SIZE))
		return EXIT_FAILURE;

#if USE_DYE
	if (!DyeImage.loadFromFile("img/201x201.png"))
	{
		std::cout << "Cannot open the specified DyeTexture" << std::endl;
		return EXIT_FAILURE;
	}
#endif

	SpriteDynamicTexture.setTexture(DynamicTexture);
	DynamicTexture.setSmooth(true);
	float SpriteScale = static_cast<float>(Canvas.getSize().x) / static_cast<float>(PLOT_RESOLUTION);
	SpriteDynamicTexture.scale(SpriteScale, SpriteScale);
	Uint64 PixelsBufferSize = PLOT_RESOLUTION * PLOT_RESOLUTION * 4;
	Uint8* Pixels = new Uint8[PixelsBufferSize];	memset(Pixels, 0, PixelsBufferSize * sizeof(Uint8));



	Vector2f WindowOffset = Vector2f(0.95f, 0.95f);
	Vector2f WindowSize = Vector2f(window.getSize().x * WindowOffset.x, window.getSize().y * WindowOffset.y);
	Vector2f ImageSize = Vector2f(static_cast<float>(Canvas.getSize().x), static_cast<float>(Canvas.getSize().y));

	Sprite FinalSprite;
	float FinalSpriteScale = (WindowSize.y) / ImageSize.y;
	FinalSprite.scale(FinalSpriteScale, -FinalSpriteScale);

	Vector2f FinalSpritePosition;
	FinalSpritePosition.x = (WindowSize.x * 0.5f) / WindowOffset.x;
	FinalSpritePosition.x -= (ImageSize.x * FinalSprite.getScale().x) * 0.5f;

	FinalSpritePosition.y = (WindowSize.y * 0.5f) / WindowOffset.y;
	FinalSpritePosition.y -= (ImageSize.y * FinalSprite.getScale().y) * 0.5f;

	FinalSprite.move(FinalSpritePosition);

#if USE_CUDA
	Uint8* Pixels_d;
	cudaMalloc((void **)&Pixels_d, PixelsBufferSize * sizeof(Uint8));		
	cudaMemset(Pixels_d, 255, PixelsBufferSize * sizeof(Uint8));
	
#if USE_DYE
	Uint8* DyeImage_d;
	cudaMalloc((void **)&DyeImage_d, PixelsBufferSize * sizeof(Uint8));		
	cudaMemcpy(DyeImage_d, DyeImage.getPixelsPtr(), PixelsBufferSize * sizeof(Uint8), cudaMemcpyHostToDevice);
#endif
#endif
#else
	RenderWindow window(VideoMode(480, 320), "Navier Stokes");
#endif
	
	// --------------------------------------------------------------------------------
	// Set initial parameters
	Real Viscocity = 1.0f / REYNOLDS_NUMBER;

	// relaxation factors
	// SOR converges fastest for a square lattice if Beta = 2 / (1.0f + PI / N)
	// where N is the number of lattice points in the x or y directions
	Real Beta = 2.0f / (1.0f + (PI / GRID_SIZE));

	// Spatial step
	Real h = 1.0f / (Real)(GRID_SIZE - 1);

	// --------------------------------------------------------------------------------
	// Data Buffers
	Uint32 SizeOfData = sizeof(Real)* GRID_SIZE * GRID_SIZE;
	Real * u =		new Real[GRID_SIZE * GRID_SIZE];		memset(u, 0, SizeOfData);
	Real * v =		new Real[GRID_SIZE * GRID_SIZE];		memset(v, 0, SizeOfData);
	Real * max	=	new Real[GRID_SIZE * GRID_SIZE];		memset(v, 0, SizeOfData);
	Real * phi =	new Real[GRID_SIZE * GRID_SIZE];		memset(phi, 0, SizeOfData);
	Real * omega =	new Real[GRID_SIZE * GRID_SIZE];		memset(omega, 0, SizeOfData);
	Real * w =		new Real[GRID_SIZE * GRID_SIZE];		memset(w, 0, SizeOfData);

	Particle Particles[NUM_PARTICLES];
	srand(static_cast <unsigned> (time(0)));

	for (Uint32 i = 0; i < NUM_PARTICLES; i++)
	{
		float r_i = static_cast <float>(rand()) / static_cast <float> (RAND_MAX);
		float r_j = static_cast <float>(rand()) / static_cast <float> (RAND_MAX);

		Particles[i].SetParticle(
			Vector2f(FinalSpritePosition.x, FinalSpritePosition.y - WindowSize.y),
			Vector2f(window.getSize().x - FinalSpritePosition.x, FinalSpritePosition.y),
			GRID_SIZE * r_i, GRID_SIZE * r_j,
			u, v);
	}

#if USE_CUDA
	// --------------------------------------------------------------------------------
	// CUDA Device Data Buffers
	DeviceQuery();

	int CudaDevice = 0;
	cudaSetDevice(CudaDevice);
	cudaDeviceProp CudaDeviceProp;
	cudaGetDeviceProperties(&CudaDeviceProp, CudaDevice);

	Real * u_d;
	Real * v_d;
	Real * max_d;
	Real * phi_d;
	Real * omega_d;
	Real * w_d;

	cudaMalloc((void **)&u_d, SizeOfData);		cudaMemset(u_d, 0, SizeOfData);
	cudaMalloc((void **)&v_d, SizeOfData);		cudaMemset(v_d, 0, SizeOfData);
	cudaMalloc((void **)&max_d, SizeOfData);	cudaMemset(max_d, 0, SizeOfData);
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
				CopyDataFromDeviceToHost(omega, omega_d, u, u_d, v, v_d, phi, phi_d, w, w_d);
#endif
				std::stringstream ss_u;
				WriteArray(GRID_SIZE, u, SimulationTime, "Data/u.csv");

				std::stringstream ss_v;
				WriteArray(GRID_SIZE, v, SimulationTime, "Data/v.csv");

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
		for (Uint64 it = 0; it < 1; it++)
		{
			SOR(omega_d, phi_d, w_d, h, Beta, CudaDeviceProp);
		}

		Real dt = UpdateVorticity(omega_d, u_d, v_d, max_d, max, phi_d, w_d, h, Viscocity, CudaDeviceProp);

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
					{
						phi[IJ(i, j)] = 0.25f*Beta*(
								phi[IJ(i + 1, j)] + phi[IJ(i - 1, j)] + 
								phi[IJ(i, j + 1)] + phi[IJ(i, j - 1)] + 
								h*h * omega[IJ(i, j)]) 
							+ (1.0f - Beta)*phi[IJ(i, j)];
					}
					// Estimate tolerance error
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
			omega[IJ(GRID_SIZE - 1, i)] = -2.0f*phi[IJ(GRID_SIZE - 2, i)] / (h*h); // right wall
			omega[IJ(0, i)] = -2.0f*phi[IJ(1, i)] / (h*h); // left wall
		}

		// --------------------------------------------------------------------------
		// RHS Calculation
		// u+v
		Real u_v = -999999;

		for (int i = 1; i < GRID_SIZE - 1; i++)
		{
			for (int j = 1; j < GRID_SIZE - 1; j++)
			{
				u[IJ(i, j)] =  (phi[IJ(i, j + 1)] - phi[IJ(i, j - 1)]) / (2 * h);
				v[IJ(i, j)] = -(phi[IJ(i + 1, j)] - phi[IJ(i - 1, j)]) / (2 * h);
				Real sum = u[IJ(i, j)] + v[IJ(i, j)];
				max[IJ(i, j)] = sum;
				if (sum > u_v)
					u_v = sum;

				w[IJ(i, j)] = -0.25f*((phi[IJ(i, j + 1)] - phi[IJ(i, j - 1)])*(omega[IJ(i + 1, j)] - omega[IJ(i - 1, j)])
					- (phi[IJ(i + 1, j)] - phi[IJ(i - 1, j)])*(omega[IJ(i, j + 1)] - omega[IJ(i, j - 1)])) / (h*h)
					+ Viscocity * (omega[IJ(i + 1, j)] + omega[IJ(i - 1, j)] + omega[IJ(i, j + 1)] + omega[IJ(i, j - 1)] - 4.0f*omega[IJ(i, j)]) / (h*h);
			}
		}

		// -------------------------------------------------------------------------
		// Get an apropiate dt
		Real dt = (8 * REYNOLDS_NUMBER * h * h) / (16 + u_v * u_v * REYNOLDS_NUMBER * REYNOLDS_NUMBER * h * h);

		// -------------------------------------------------------------------------
		// Update the vorticity
		for (int i = 1; i < GRID_SIZE - 1; i++)
		{
			for (int j = 1; j < GRID_SIZE - 1; j++)
				omega[IJ(i, j)] = omega[IJ(i, j)] + dt * w[IJ(i, j)];
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
		SimulationTime += dt;
		CurrentStep++;
#endif
#if PRINT_DATA
		{
			std::cout
				<< "Sim dt: " << dt << "\t"
				<< "Current step: " << CurrentStep << "\t"
				<< "Step time: " << Clock1.restart().asSeconds() << " sec\t"
				<< termcolor::red 
				<< "Simulation time: " << SimulationTime << " sec\t"
				<< termcolor::reset
				<< "Elapsed time: " << Clock2.getElapsedTime().asSeconds() << " sec\n" << std::endl;
		}
#endif //PRINT_DATA
#if USE_CPP_PLOT
		Canvas.clear(Color::Black);
#if USE_CUDA

		CopyDataFromDeviceToHost(phi, phi_d);
		CopyDataFromDeviceToHost(u, u_d);
		CopyDataFromDeviceToHost(v, v_d);
#endif
		Plot(GRID_SIZE, Pixels, v);
		DynamicTexture.update(Pixels);
		Canvas.draw(SpriteDynamicTexture);
		Canvas.display();

		// --------------------------------------------------------
		// Draw the final image result
		window.clear(Color::Black);
		FinalSprite.setTexture(Canvas.getTexture());
		window.draw(FinalSprite);

		for (Uint32 i = 0; i < NUM_PARTICLES; i++)
			Particles[i].Update(window, dt);

		// end the current frame
		window.display();
#endif
	}

#if USE_CUDA
	cudaFree(u_d);
	cudaFree(v_d);
	cudaFree(max_d);
	cudaFree(phi_d);
	cudaFree(omega_d);
	cudaFree(w_d);
#endif
#if USE_CPP_PLOT
	delete[] Pixels;
#if USE_CUDA
	cudaFree(Pixels_d);
#endif
#endif
	delete[] u;
	delete[] v;
	delete[] max;
	delete[] phi;
	delete[] omega;
	delete[] w;

	return 0;
}