#include <iostream>
#include <stdio.h>
#include <SFML/Graphics.hpp>
#include <mgl2/mgl.h>
#include "Functions.h"


using namespace sf;

void SimpleDataFill(Uint8* Pixels, Uint64 Dimensions)
{
	for (Uint64 i = 0; i < Dimensions; i+=4)
	{
		Uint8 Color = (i * 255) / (float)Dimensions;

		Pixels[i] = Color;
		Pixels[i + 1] = Color;
		Pixels[i + 2] = Color;
		Pixels[i+3] = 255;
	}
}

void mgls_prepare2v(mglData *a, mglData *b)
{
	register long i, j, n = 20, m = 30, i0;
	if (a) a->Create(n, m);   if (b) b->Create(n, m);
	mreal x, y;
	for (i = 0; i<n; i++)  for (j = 0; j<m; j++)
	{
		x = i / (n - 1.); y = j / (m - 1.); i0 = i + n*j;
		if (a) 
			mgl_data_set_value(a, i0,0.6*sin(2 * M_PI*x)*sin(3 * M_PI*y) + 0.4*cos(3 * M_PI*x*y), 0, 0);
		if (b) 
			mgl_data_set_value(a, i0, 0.6*cos(2 * M_PI*x)*cos(3 * M_PI*y) + 0.4*cos(3 * M_PI*x*y), 0, 0);
	}
}

void Fill(mglData *a, const Number * InData)
{
	register long i, j, n = N, m = N, i0;

	if (a)
		mgl_data_create(a, n, m, 1);

	mreal x, y;
	for (i = 0; i<n; i++)  for (j = 0; j<m; j++)
	{
		if (a)
			mgl_data_set_value(a, InData[IJ(i, j)], i, j, 0);
	}
}

mglData a, b;
int sample(mglGraph *gr, Uint8* Pixels, Uint64 PixelsBufferSize, const Number * InData)
{
#if 0
	gr->SubPlot(2, 2, 0); gr->Title("Box (default)"); gr->Rotate(50, 60);
	gr->Box();
	gr->SubPlot(2, 2, 1); gr->Title("colored");   gr->Rotate(50, 60);
	gr->Box("r");
	gr->SubPlot(2, 2, 2); gr->Title("with faces");  gr->Rotate(50, 60);
	gr->Box("@");
	gr->SubPlot(2, 2, 3); gr->Title("both");  gr->Rotate(50, 60);
	gr->Box("@cm");
#else
	gr->ClearFrame();
	Fill(&a, InData);
	gr->Box();
	gr->Dens(a, "BbcyrR");
	gr->Cont(a, "b");
#endif
	//=================================================
	const Uint8 * data = mgl_get_rgba(gr->Self());
	memcpy(Pixels, data, PixelsBufferSize * sizeof(Uint8));

	return 0;
}

int main(int argc, char **argv)
{
	RenderWindow window(VideoMode(800, 600), "Navier Stokes");

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
	int NumberOfIteerations = 0;
	int nx = N;
	int ny = N;
	Number ReynoldsNumber = 10.0f;
	Number Viscocity = 1.0f / ReynoldsNumber;
	Number dt = 0.02f;
	Number LidSpeed = 1.0f;

	// number of iterations
	int no_it = 100;
	// relaxation factors
	Number Beta = 1.5f;
	// parameter for SOR iteration
	Number MaxErr = 0.001f;

	Number h = 1.0f / (Number)(nx - 1);

	float t = 0.0f;

	// --------------------------------------------------------------------------------
	// Data Buffers
	Uint32 SizeOfData = sizeof(Number) * nx * ny;
	Uint64 PixelsBufferSize = PLOT_RESOLUTION * PLOT_RESOLUTION * 4;
	mglGraph			gr(0, PLOT_RESOLUTION, PLOT_RESOLUTION);
	Uint8* Pixels =		new Uint8[PixelsBufferSize];		memset(Pixels, 0, PixelsBufferSize * sizeof(Uint8));
	SimpleDataFill(Pixels, PixelsBufferSize);
	Number * phi =		new Number[nx * ny];				memset(phi, 0, SizeOfData);
	Number * omega =	new Number[nx * ny];				memset(omega, 1, SizeOfData);
	Number * w =		new Number[nx * ny];				memset(w, 0, SizeOfData);
	
	// run the program as long as the window is open
	bool bSimulateNextFrame = false;
	bool bUseKeyToSimulate = true;
	Uint64 iteration = 0;

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
		}

		if (bUseKeyToSimulate && !bSimulateNextFrame)
			continue;

		NumberOfIteerations++;

#if 1 // Stream-Vorticity Calculation
		if (NumberOfIteerations <= 60)
		{
			Canvas.clear(Color::Black);

			// -------------------------------------------------------------------------
			// streamfunction calculation by SOR iteration
			for (int it = 0; it < no_it; it++)
			{
				memcpy(w, phi, SizeOfData);

				for (int i = 1; i < nx - 1; i++)
				{
					for (int j = 1; j < ny - 1; j++)
						phi[IJ(i, j)] = 0.25f*Beta*(phi[IJ(i + 1, j)] + phi[IJ(i - 1, j)] + phi[IJ(i, j + 1)] + phi[IJ(i, j - 1)] + h*h*omega[IJ(i, j)]) + (1.0f - Beta)*phi[IJ(i, j)];
				}

				Number Err = 0.0f;

				for (int i = 0; i < nx; i++)
				{
					for (int j = 0; j < ny; j++)
						Err += abs(w[IJ(i, j)] - phi[IJ(i, j)]);
				}

				// stop if iteration has converged
				if (Err <= MaxErr)
					break;
			}
			// -------------------------------------------------------------------------
			// boundary conditions for the Vorticity
			for (int i = 0; i < nx; i++)
			{
				for (int j = 0; j < ny; j++)
				{
					omega[IJ(i, 0)] = -2.0f*phi[IJ(i, 1)] / (h*h); // bottom wall
					omega[IJ(i, ny - 1)] = -2.0f*phi[IJ(i, ny - 2)] / (h*h) - LidSpeed * (2.0f / h); // top wall
					omega[IJ(0, j)] = -2.0f*phi[IJ(1, j)] / (h*h); // right wall
					omega[IJ(nx - 1, j)] = -2.0f*phi[IJ(nx - 2, j)] / (h*h); // left wall
				}
			}
			// --------------------------------------------------------------------------
			// RHS Calculation
			for (int i = 1; i < nx - 1; i++)
			{
				for (int j = 1; j < ny - 1; j++)
				{
					w[IJ(i, j)] = -0.25f*((phi[IJ(i, j + 1)] - phi[IJ(i, j - 1)])*(omega[IJ(i + 1, j)] - omega[IJ(i - 1, j)])
						- (phi[IJ(i + 1, j)] - phi[IJ(i - 1, j)])*(omega[IJ(i, j + 1)] - omega[IJ(i, j - 1)])) / (h*h)
						+ Viscocity * (omega[IJ(i + 1, j)] + omega[IJ(i - 1, j)] + omega[IJ(i, j + 1)] + omega[IJ(i, j - 1)] - 4.0f*omega[IJ(i, j)]) / (h*h);
				}
			}
			// -------------------------------------------------------------------------
			// Update the vorticity
			for (int i = 1; i < nx - 1; i++)
			{
				for (int j = 1; j < ny - 1; j++)
					omega[IJ(i, j)] = omega[IJ(i, j)] + dt*w[IJ(i, j)];
			}

			// increment the time
			t = t + dt;

		}
#endif
		sample(&gr, Pixels, PixelsBufferSize, omega);
		DynamicTexture.update(Pixels);
		Canvas.draw(SpriteDynamicTexture);
		Canvas.display();

#if 1 
		// --------------------------------------------------------
		// Draw the final image result
		window.clear(Color::Green);

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

		std::cout << clock.getElapsedTime().asSeconds() << std::endl;
		clock.restart();

#endif
	}

	delete[] Pixels;
	delete[] omega;
	delete[] phi;
	delete[] w;

	return 0;
}