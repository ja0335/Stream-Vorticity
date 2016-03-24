#include "Functions.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <limits>

using namespace sf;
using namespace std;

void WriteArray(sf::Uint32 GridSize, const float * ArayToDebug)
{
	ofstream myfile;
	myfile.open("array.csv");

	for (Uint32 j = 0; j < GridSize; j++)
	{
		for (Uint32 i = 0; i < GridSize; i++)
		{
			int Index = IJ(i, j);
			float value = ArayToDebug[Index];
			myfile << value << ",";
		}

		myfile << "\n";
	}

	myfile.close();
}

void DebugArray(sf::Uint32 GridSize, const float * ArayToDebug, sf::Uint8* Pixels)
{
	float HighestValue = numeric_limits<float>::min();
	float LowestValue = numeric_limits<float>::max();

	for (Uint32 i = 0; i < GridSize; i++)
	{
		for (Uint32 j = 0; j < GridSize; j++)
		{
			int Index = IJ(i, j);
			if (ArayToDebug[Index] > HighestValue)
				HighestValue = ArayToDebug[Index];
			if (ArayToDebug[Index] < LowestValue)
				LowestValue = ArayToDebug[Index];
		}
	}

	HighestValue = abs(LowestValue) + HighestValue;

	for (Uint32 j = 0; j < GridSize; j++)
	{
		for (Uint32 i = 0; i < GridSize; i++)
		{
			Uint64 Index = IJ(i, j);
			float NewValue = abs(LowestValue) + ArayToDebug[Index];
			Uint8 Color = static_cast<Uint8>((NewValue * 255.0f) / HighestValue);

			Pixels[Index * 4 + 0] = Color;
			Pixels[Index * 4 + 1] = Color;
			Pixels[Index * 4 + 2] = Color;
			Pixels[Index * 4 + 3] = 255;
		}
	}
}

void SetBoundary(sf::Uint32 GridSize, float Value, EWall Wall, float * InArray)
{
	int Index = 0;

	for (Uint32 i = 0; i < GridSize; i++)
	{
		if (Wall == EWall::EWall_Top)
			Index = IJ(i, 0);
		else if (Wall == EWall::EWall_Bottom)
			Index = IJ(i, GridSize - 1);
		else if (Wall == EWall::EWall_Left)
			Index = IJ(GridSize - 1, i);
		else if (Wall == EWall::EWall_Right)
			Index = IJ(0, i);

		InArray[Index] = Value;
	}
}

void SetVelocityColor(Uint32 GridSize, const float* u, const float* v, float& HighestSqNorm, Uint8* Pixels)
{
	for (Uint32 i = 0; i < GridSize; i++)
	{
		for (Uint32 j = 0; j < GridSize; j++)
		{
			Uint64 ij = IJ(i, j);
			float SqNorm = u[ij] * u[ij] + v[ij] * v[ij];
			Uint8 Color = static_cast<Uint8>((SqNorm * 255.0f) / HighestSqNorm);

			Pixels[ij * 4 + 0] = Color;
			Pixels[ij * 4 + 1] = Color;
			Pixels[ij * 4 + 2] = Color;
			Pixels[ij * 4 + 3] = 255;
		}
	}
}

void DrawVelocities(sf::Uint32 GridSize, float SpriteScale, const float* u, const float* v, sf::RenderTexture& Canvas)
{
	//float HighestSqNorm = 0.0f;

	//for (Uint32 j = 0; j < GridSize; j ++)
	//{
	//	for (Uint32 i = 0; i < GridSize; i++)
	//	{
	//		float SqNorm = u[IJ(i, j)] * u[IJ(i, j)] + v[IJ(i, j)] * v[IJ(i, j)];

	//		if (SqNorm > HighestSqNorm)
	//			HighestSqNorm = SqNorm;
	//	}
	//}

	Uint32 Spacing = 2; //GridSize / HighestSqNorm;

	for (Uint32 j = 0; j < GridSize; j += Spacing)
	{
		for (Uint32 i = 0; i < GridSize; i += Spacing)
		{
			VertexArray triangle(Lines, 2);
			triangle[1].position = Vector2f(i * SpriteScale, j * SpriteScale);
			triangle[0].position = Vector2f((i + u[IJ(i, j)]) * SpriteScale, (j + v[IJ(i, j)]) * SpriteScale);

			triangle[0].color = Color::Red;
			triangle[1].color = Color::Red;

			Canvas.draw(triangle);
		}
	}
}

void ProduceVorticity(Uint32 GridSize, float dx, float dy, const float* u, const float* v, float* omega)
{
	// Interior grid points
	for (Uint32 j = 1; j < GridSize - 1; j++)
	{
		for (Uint32 i = 1; i < GridSize - 1; i++)
			omega[IJ(i, j)] = -(u[IJ(i, j + 1)] - u[IJ(i, j - 1)]) / (2 * dy) + (v[IJ(i + 1, j)] - v[IJ(i - 1, j)]) / (2 * dx);
	}

	// Top wall
	for (Uint32 i = 0; i < GridSize; i++)
		omega[IJ(i, 0)] = (u[IJ(i, 1)] - u[IJ(i, 0)]) / dy;

	// Bottom wall
	for (Uint32 i = 0; i < GridSize; i++)
		omega[IJ(i, GridSize - 1)] = (u[IJ(i, GridSize - 1)] - u[IJ(i, GridSize - 2)]) / dy;

	// Left wall
	for (Uint32 j = 0; j < GridSize; j++)
		omega[IJ(0, j)] = (v[IJ(1, j)] - v[IJ(0, j)]) / dx;

	// Right wall
	for (Uint32 j = 0; j < GridSize; j++)
		omega[IJ(GridSize - 1, j)] = (v[IJ(GridSize - 1, j)] - v[IJ(GridSize - 2, j)]) / dx;
}

void IntegrateVorticity(sf::Uint32 GridSize, float ReynoldsNumber, float dx, float dy, float dt, const float* u, const float* v, float* omega, float* helper_omega)
{
	for (Uint32 j = 1; j < GridSize - 1; j++)
	{
		for (Uint32 i = 1; i < GridSize - 1; i++)
		{
			float Laplacian = (omega[IJ(i + 1, j)] - 2 * omega[IJ(i, j)] + omega[IJ(i - 1, j)]) / (dx*dx)
				+ (omega[IJ(i, j + 1)] - 2 * omega[IJ(i, j)] + omega[IJ(i, j - 1)]) / (dy*dy);

			helper_omega[IJ(i, j)] = omega[IJ(i, j)] +
				dt * (-u[IJ(i, j)] * ((omega[IJ(i + 1, j)] - omega[IJ(i - 1, j)]) / (2 * dx))
				- v[IJ(i, j)] * ((omega[IJ(i, j + 1)] - omega[IJ(i, j - 1)]) / (2 * dy))
				+ (1.0f / ReynoldsNumber) * Laplacian);
		}
	}

	memcpy(omega, helper_omega, GridSize * GridSize * sizeof(float));
}

void IntegrateStream(sf::Uint32 GridSize, float dx, float dy, float LidVelocity, const float* omega, float * psi, float * helper)
{
	for (Uint32 sor = 0; sor < 50; sor++)
	{
		memcpy(helper, psi, GridSize * GridSize * sizeof(float));

		for (Uint32 j = 1; j < GridSize - 1; j++)
		{
			for (Uint32 i = 1; i < GridSize - 1; i++)
			{
				psi[IJ(i, j)] = (dy*dy * (helper[IJ(i + 1, j)] + helper[IJ(i - 1, j)]) + dx*dx *(helper[IJ(i, j + 1)] + helper[IJ(i, j - 1)]) - omega[IJ(i, j)] * dx*dx*dy*dy) / (2 * (dx*dx + dy*dy));
			}
		}
	}
}

void CalculateVelocity(sf::Uint32 GridSize, float dx, float dy, float LidVelocity, const float * psi, float * u, float * v)
{
	SetBoundary(N, LidVelocity, EWall::EWall_Top, u);
	SetBoundary(N, 0.0f, EWall::EWall_Bottom, u);
	SetBoundary(N, 0.0f, EWall::EWall_Left, u);
	SetBoundary(N, 0.0f, EWall::EWall_Right, u);

	SetBoundary(N, 0.0f, EWall::EWall_Top, v);
	SetBoundary(N, 0.0f, EWall::EWall_Bottom, v);
	SetBoundary(N, 0.0f, EWall::EWall_Left, v);
	SetBoundary(N, 0.0f, EWall::EWall_Right, v);

	for (Uint32 j = 1; j < GridSize - 1; j++)
	{
		for (Uint32 i = 1; i < GridSize - 1; i++)
		{
			u[IJ(i, j)] = (psi[IJ(i, j + 1)] - psi[IJ(i, j - 1)]) / (2 * dy);
			v[IJ(i, j)] = (psi[IJ(i - 1, j)] - psi[IJ(i + 1, j)]) / (2 * dx);
		}
	}
}

void PresureStabilization(sf::Uint32 GridSize, float rho, float dx, float dy, float dt, const float * u, const float * v, float * p, float * helper)
{
	for (Uint32 sor = 0; sor < 50; sor++)
	{
		memcpy(helper, p, GridSize * GridSize * sizeof(float));

		for (Uint32 j = 1; j < GridSize - 1; j++)
		{
			for (Uint32 i = 1; i < GridSize - 1; i++)
			{
				p[IJ(i, j)] = (dy*dy * (helper[IJ(i + 1, j)] + helper[IJ(i - 1, j)]) + dx*dx *(helper[IJ(i, j + 1)] + helper[IJ(i, j - 1)])) / (2 * (dx*dx + dy*dy))
					- (rho*(dx*dx)*(dy*dy)) / (2 * (dx*dx + dy*dy)) *
					(1 / dt * ((u[IJ(i + 1, j)] - u[IJ(i - 1, j)]) / (2 * dx) + (v[IJ(i, j + 1)] - v[IJ(i, j - 1)]) / (2 * dy)) -
					((u[IJ(i + 1, j)] - u[IJ(i - 1, j)]) / (2 * dx)) * ((u[IJ(i + 1, j)] - u[IJ(i - 1, j)]) / (2 * dx)) -
					2 * ((u[IJ(i, j + 1)] - u[IJ(i, j - 1)]) / (2 * dy) * (v[IJ(i + 1, j)] - v[IJ(i - 1, j)]) / (2 * dx)) -
					((v[IJ(i, j + 1)] - v[IJ(i, j - 1)]) / (2 * dy)) * ((v[IJ(i, j + 1)] - v[IJ(i, j - 1)]) / (2 * dy))
					);
			}
		}

		for (Uint32 i = 0; i < GridSize; i++)
		{
			//p[:,-1] =p[:,-2] ##dp/dy = 0 at x = 2
			p[IJ(i, 0)] = p[IJ(i, 1)];

			//p[0,:] = p[1,:]  ##dp/dy = 0 at y = 0
			p[IJ(i, GridSize - 1)] = p[IJ(i, GridSize - 2)];

			//p[:,0]=p[:,1]    ##dp/dx = 0 at x = 0
			p[IJ(0, i)] = p[IJ(1, i)];

			//p[-1,:]=0        ##p = 0 at y = 2
			p[IJ(i, 0)] = 0;
		}
	}
}

void UpdateVelocity(sf::Uint32 GridSize, float dx, float dy, float dt, float LidVelocity, float rho, float nu, const float * p, float * u, float * v, float * un, float * vn)
{
	memcpy(un, u, GridSize * GridSize * sizeof(float));
	memcpy(vn, v, GridSize * GridSize * sizeof(float));

	for (Uint32 j = 1; j < GridSize - 1; j++)
	{
		for (Uint32 i = 1; i < GridSize - 1; i++)
		{
			u[IJ(i, j)] = un[IJ(i, j)] -
				(un[IJ(i, j)] * (dt / dx)) * (un[IJ(i, j)] - un[IJ(i - 1, j)]) -
				(vn[IJ(i, j)] * (dt / dy)) * (un[IJ(i, j)] - un[IJ(i, j - 1)]) -
				dt / (2 * rho*dx)*(p[IJ(i + 1, j)] - p[IJ(i - 1, j)]) +
				nu*(dt / (dx*dx))*(un[IJ(i + 1, j)] - 2 * un[IJ(i, j)] + un[IJ(i - 1, j)]) +
				nu*(dt / (dy*dy))*(un[IJ(i, j + 1)] - 2 * un[IJ(i, j)] + un[IJ(i, j - 1)]);

			v[IJ(i, j)] = vn[IJ(i, j)] -
				(un[IJ(i, j)] * (dt / dx)) * (vn[IJ(i, j)] - vn[IJ(i - 1, j)]) -
				(vn[IJ(i, j)] * (dt / dy)) * (vn[IJ(i, j)] - vn[IJ(i, j - 1)]) -
				dt / (2 * rho*dy)*(p[IJ(i, j + 1)] - p[IJ(i, j - 1)]) +
				nu*(dt / (dx*dx))*(vn[IJ(i + 1, j)] - 2 * vn[IJ(i, j)] + vn[IJ(i - 1, j)]) +
				nu*(dt / (dy*dy))*(vn[IJ(i, j + 1)] - 2 * vn[IJ(i, j)] + vn[IJ(i, j - 1)]);
		}
	}

	SetBoundary(GridSize, LidVelocity, EWall::EWall_Top, u);
	SetBoundary(GridSize, 0.0f, EWall::EWall_Top, u);
	SetBoundary(GridSize, 0.0f, EWall::EWall_Top, u);
	SetBoundary(GridSize, 0.0f, EWall::EWall_Top, u);

	SetBoundary(GridSize, 0.0f, EWall::EWall_Top, v);
	SetBoundary(GridSize, 0.0f, EWall::EWall_Top, v);
	SetBoundary(GridSize, 0.0f, EWall::EWall_Top, v);
	SetBoundary(GridSize, 0.0f, EWall::EWall_Top, v);
}