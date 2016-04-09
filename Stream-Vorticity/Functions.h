#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include <SFML/Graphics.hpp>
#include <mgl2/mgl.h>
#include "Macros.h"


void WriteArray(sf::Uint32 GridSize, const Real * ArayToDebug, Real Time, const char * Name = "array.csv");

void mgls_prepare2v(const Real * phi, Real h, mglData *a, mglData *b);

/*
* @param	DataScaleFactor,	 Scales the values in the InData
*/
void Fill(mglData *a, const Real * InData, Real DataScaleFactor = 1.0f);

/*
* @param	DataScaleFactor,	 Scales the values in the InData
*/
int Plot(mglGraph *gr, sf::Uint8* Pixels, sf::Uint64 PixelsBufferSize, const Real * InData, Real DataScaleFactor = 1.0f);

int Plot2(mglGraph *gr, sf::Uint8* Pixels, sf::Uint64 PixelsBufferSize, const Real * phi, const Real * omega);

#endif