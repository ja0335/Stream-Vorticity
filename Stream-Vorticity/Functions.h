#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include <SFML/Graphics.hpp>
#include "Macros.h"


void WriteArray(sf::Uint32 GridSize, const Real * ArayToDebug, Real Time, const char * Name = "array.csv");

void GetMinMaxValues(sf::Uint32 GridSize, const Real * InData, Real &MinValue, Real &MaxValue);

int Plot(sf::Uint32 GridSize, sf::Uint8* Pixels, const Real * InData);

#endif