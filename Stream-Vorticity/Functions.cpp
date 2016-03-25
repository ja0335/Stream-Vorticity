#include "Functions.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <limits>

using namespace sf;
using namespace std;

void WriteArray(sf::Uint32 GridSize, const Real * ArayToDebug)
{
	ofstream myfile;
	myfile.open("array.csv");

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

void mgls_prepare2v(const Real * phi, Real h, mglData *a, mglData *b)
{
	register long i, j, n = N, m = N;
	if (a) a->Create(n, m);
	if (b) b->Create(n, m);

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < m; j++)
		{
			if (a)
				mgl_data_set_value(a, IJ(i, j), (phi[IJ(i, j + 1)] - phi[IJ(i, j)]) / (2 * h), 0, 0);
			if (b)
				mgl_data_set_value(b, IJ(i, j), (phi[IJ(i + 1, j)] - phi[IJ(i, j)]) / (2 * h), 0, 0);
		}
	}

	for (i = 1; i < n - 1; i++)
	{
		for (j = 1; j < m - 1; j++)
		{
			if (a)
				mgl_data_set_value(a, IJ(i, j), (phi[IJ(i, j + 1)] - phi[IJ(i, j)]) / (2 * h), 0, 0);
			if (b)
				mgl_data_set_value(b, IJ(i, j), (phi[IJ(i + 1, j)] - phi[IJ(i, j)]) / (2 * h), 0, 0);
		}
	}
}

void Fill(mglData *a, const Real * InData, Real DataScaleFactor)
{
	register long i, j;

	if (a)
		mgl_data_create(a, N, N, 1);

	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			if (a)
				mgl_data_set_value(a, InData[IJ(i, j)] * DataScaleFactor, i, j, 0);
		}
	}
}

int Plot(mglGraph *gr, Uint8* Pixels, Uint64 PixelsBufferSize, const Real * InData, Real DataScaleFactor)
{
	mglData a, b;
	gr->ClearFrame();
	Fill(&a, InData, DataScaleFactor);
	gr->Box();
	gr->Dens(a, "BbcyrR");
	gr->Cont(a, "rb");
	//=================================================
	const Uint8 * data = mgl_get_rgba(gr->Self());
	memcpy(Pixels, data, PixelsBufferSize * sizeof(Uint8));

	return 0;
}

int Plot2(mglGraph *gr, Uint8* Pixels, Uint64 PixelsBufferSize, const Real * phi, const Real * omega)
{
	mglData a, b;
	gr->ClearFrame();

	Fill(&a, phi, 10000.0f);
	gr->SubPlot(2, 2, 0);
	gr->Box();
	gr->Dens(a, "BbcyrR");
	gr->Cont(a, "rb", "21");

	Fill(&b, omega, 1);
	gr->SubPlot(2, 2, 1);
	gr->Box();
	gr->Dens(b, "BbcyrR");
	gr->Cont(b, "rb");

	//=================================================
	const Uint8 * data = mgl_get_rgba(gr->Self());
	memcpy(Pixels, data, PixelsBufferSize * sizeof(Uint8));

	return 0;
}