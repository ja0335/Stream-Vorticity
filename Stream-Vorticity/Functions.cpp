#include "Functions.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <limits>

using namespace sf;
using namespace std;

void WriteArray(sf::Uint32 GridSize, const Real * ArayToDebug, Real Time, const char * Name)
{
	ofstream myfile;
	myfile.open(Name);

	myfile << "GRID_SIZE=" << GRID_SIZE
		<< "; REYNOLDS_NUMBER=" << REYNOLDS_NUMBER
		<< "; DT=" << DT
		<< "; TIME=" << Time
		<< std::endl;

	for (Uint32 j = 0; j < GridSize; j++)
	{
		for (Uint32 i = 0; i < GridSize; i++)
		{
			int Index = IJ(i, j);
			Real value = ArayToDebug[Index];
			myfile << value << ";";
		}

		myfile << "\n";
	}

	myfile.close();
}

void WriteVector(const std::vector<Real>* Vector, const char * Name)
{
	ofstream myfile;
	myfile.open(Name);

	myfile << "Time;Average" << std::endl;

	sf::Uint64 VectorSize = (*Vector).size();
	
	for (Uint64 i = 0; i < VectorSize; i++)
	{
		myfile << DT * AVERAGE_EACH_STEPS * i << ";" << (*Vector)[i];

		myfile << "\n";
	}

	myfile.close();
}

void Plot(sf::Uint32 GridSize, sf::Uint8* Pixels, const Real * omega)
{
	for (Uint64 i = 0; i < GridSize; i++)
	{
		for (Uint64 j = 0; j < GridSize; j += 4)
		{
			if (omega[IJ(i, j)] >= 0.0f)
			{
				Pixels[i * GridSize + (j + 0)] = 0;
				Pixels[i * GridSize + (j + 1)] = 0;
				Pixels[i * GridSize + (j + 2)] = 255;
				Pixels[i * GridSize + (j + 3)] = 255;
			}
			else
			{
				Pixels[i * GridSize + (j + 0)] = 255;
				Pixels[i * GridSize + (j + 1)] = 255;
				Pixels[i * GridSize + (j + 2)] = 255;
				Pixels[i * GridSize + (j + 3)] = 255;
			}
		}
	}
}