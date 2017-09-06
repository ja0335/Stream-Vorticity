#include "Functions.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <limits>

using namespace sf;
using namespace std;

void WriteArray(const Real * ArayToDebug, const sf::Uint32 ArraySize, const char * Name)
{
	ofstream myfile;
	myfile.open(Name);

	for (Uint32 i = 0; i < ArraySize; i++)
	{
		Real value = ArayToDebug[i];
		myfile << value << ";";
	}

	myfile.close();
}

void WriteArray(sf::Uint32 GridSize, const Real * ArayToDebug, Real Time, const char * Name)
{
	ofstream myfile;
	myfile.open(Name);

	myfile << "GRID_SIZE=" << GRID_SIZE
		<< "; REYNOLDS_NUMBER=" << REYNOLDS_NUMBER
		<< "; DT=" << "Calculated" //DT
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

void GetMinMaxValues(sf::Uint32 GridSize, const Real * InData, Real &MinValue, Real &MaxValue)
{
	MinValue = numeric_limits<Real>::max();
	MaxValue = numeric_limits<Real>::lowest();
	sf::Uint32 SizeOfData = GridSize * GridSize;

	for (Uint64 i = 0; i < SizeOfData; i++)
	{
		if (InData[i] < MinValue)
			MinValue = InData[i];
		if (InData[i] > MaxValue)
			MaxValue = InData[i];
	}
}

int Plot(sf::Uint32 GridSize, sf::Uint8* Pixels, const Real * InData)
{
	Real MinValue, MaxValue;
	GetMinMaxValues(GridSize, InData, MinValue, MaxValue);
	sf::Uint32 SizeOfData = GridSize * GridSize;

	MaxValue -= MinValue;
	Real ScaleFactor = 1;
	MaxValue *= ScaleFactor;

	for (Uint64 i = 0; i < SizeOfData; i++)
	{
		Real HelperValue = InData[i] - MinValue;
		HelperValue = (HelperValue * ScaleFactor * 255) / MaxValue;

		Pixels[i * 4 + 0] = static_cast<sf::Uint32>(HelperValue);
		Pixels[i * 4 + 1] = static_cast<sf::Uint32>(HelperValue);
		Pixels[i * 4 + 2] = static_cast<sf::Uint32>(HelperValue);
		Pixels[i * 4 + 3] = 255;
	}

	return 0;
}