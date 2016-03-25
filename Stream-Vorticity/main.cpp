#include <iostream>
#include "termcolor.h"
#include <stdio.h>
#include <SFML/Graphics.hpp>
#include <mgl2/mgl.h>
#include "Functions.h"


using namespace sf;


int main(int argc, char **argv)
{
	RenderWindow window(VideoMode(1920, 1080), "Navier Stokes");

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

	Clock clock;

	// --------------------------------------------------------------------------------
	// Set initial parameters
	Real Viscocity = 1.0f / REYNOLDS_NUMBER;
	Real LidSpeed = 1.0f;
		
	// relaxation factors
	// SOR converges fastest for a square lattice if Beta = 2 / (1.0f + PI / L)
	// where  is the number of lattice points in the x or y directions
	Real Beta = 2.0f / (1.0f + (PI / N));
	
	// Spatial step
	Real h = 1.0f / (Real)(N-1);
	
	// --------------------------------------------------------------------------------
	// Data Buffers
	Uint32 SizeOfData = sizeof(Real) * N * N;
	Uint64 PixelsBufferSize = PLOT_RESOLUTION * PLOT_RESOLUTION * 4;
	mglGraph			gr(0, PLOT_RESOLUTION, PLOT_RESOLUTION);
	Uint8* Pixels =		new Uint8[PixelsBufferSize];	memset(Pixels, 0, PixelsBufferSize * sizeof(Uint8));
	Real * phi =		new Real[N * N];				memset(phi, 0, SizeOfData);
	Real * omega =	new Real[N * N];				memset(omega, 0, SizeOfData);
	Real * w =		new Real[N * N];				memset(w, 0, SizeOfData);
	
	// run the program as long as the window is open
	bool bSimulateNextFrame = false;
	bool bUseKeyToSimulate = true;
	Uint64 CurrentStep = 1;
	Real SimulationTime = 0.0f;
	Real RealSimulationTime = 0.0f;

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
				WriteArray(N, phi);
		}

		if (bUseKeyToSimulate && !bSimulateNextFrame)
			continue;
		
#if 1 // Stream-Vorticity Calculation
		Canvas.clear(Color::Black);

		// -------------------------------------------------------------------------
		// streamfunction calculation by SOR iteration
		for (int it = 0; it < MAXSOR_ITERATIONS; it++)
		{
			memcpy(w, phi, SizeOfData);

			for (int i = 1; i < N - 1; i++)
			{
				for (int j = 1; j < N - 1; j++)
					phi[IJ(i, j)] = 0.25f*Beta*(phi[IJ(i + 1, j)] + phi[IJ(i - 1, j)] + phi[IJ(i, j + 1)] + phi[IJ(i, j - 1)] + h*h*omega[IJ(i, j)]) + (1.0f - Beta)*phi[IJ(i, j)];
			}

			Real Err = 0.0f;

			for (int i = 0; i < N; i++)
			{
				for (int j = 0; j < N; j++)
					Err += abs(w[IJ(i, j)] - phi[IJ(i, j)]);
			}

			// stop if iteration has converged
			if (Err <= SOR_TOLERANCE_ERROR)
				break;
		}
		// -------------------------------------------------------------------------
		// boundary conditions for the Vorticity
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				omega[IJ(i, 0)] = -2.0f*phi[IJ(i, 1)] / (h*h); // bottom wall
				omega[IJ(i, N - 1)] = -2.0f*phi[IJ(i, N - 2)] / (h*h) - LidSpeed * (2.0f / h); // top wall
				omega[IJ(0, j)] = -2.0f*phi[IJ(1, j)] / (h*h); // right wall
				omega[IJ(N - 1, j)] = -2.0f*phi[IJ(N - 2, j)] / (h*h); // left wall
			}
		}
		// --------------------------------------------------------------------------
		// RHS Calculation
		for (int i = 1; i < N - 1; i++)
		{
			for (int j = 1; j < N - 1; j++)
			{
				w[IJ(i, j)] = -0.25f*((phi[IJ(i, j + 1)] - phi[IJ(i, j - 1)])*(omega[IJ(i + 1, j)] - omega[IJ(i - 1, j)])
					- (phi[IJ(i + 1, j)] - phi[IJ(i - 1, j)])*(omega[IJ(i, j + 1)] - omega[IJ(i, j - 1)])) / (h*h)
					+ Viscocity * (omega[IJ(i + 1, j)] + omega[IJ(i - 1, j)] + omega[IJ(i, j + 1)] + omega[IJ(i, j - 1)] - 4.0f*omega[IJ(i, j)]) / (h*h);
			}
		}
		// -------------------------------------------------------------------------
		// Update the vorticity
		for (int i = 1; i < N - 1; i++)
		{
			for (int j = 1; j < N - 1; j++)
				omega[IJ(i, j)] = omega[IJ(i, j)] + DT*w[IJ(i, j)];
		}

		// increment the time
		SimulationTime += DT;
		RealSimulationTime += clock.getElapsedTime().asSeconds();
#endif

		if (SimulationTime >= 1.5f)
		//if(CurrentStep >= 30)
		{
			Plot2(&gr, Pixels, PixelsBufferSize, phi, omega);

			DynamicTexture.update(Pixels);
			Canvas.draw(SpriteDynamicTexture);
			Canvas.display();

			std::cout << termcolor::bold << termcolor::green
				<< "Current step: " << CurrentStep << "\t"
				<< "Step time: " << clock.getElapsedTime().asSeconds() << "\t"
				<< "Simulation time: " << SimulationTime << "\t"
				<< "Elapsed time: " << RealSimulationTime << "\n" << std::endl;
		}
		else
		{
			std::cout << termcolor::reset
				<< "Current step: " << CurrentStep << "\t"
				<< "Step time: " << clock.getElapsedTime().asSeconds() << "\t"
				<< "Simulation time: " << SimulationTime << "\t"
				<< "Elapsed time: " << RealSimulationTime << "\n" << std::endl;
		}

#if 1 
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
		clock.restart();

#endif
		
		CurrentStep++;
	}

	delete[] Pixels;
	delete[] omega;
	delete[] phi;
	delete[] w;

	return 0;
}