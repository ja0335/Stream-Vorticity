#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include <vector>
#include <SFML/Graphics.hpp>
#include "Macros.h"

void WriteArray(sf::Uint32 GridSize, const Real * ArayToDebug, Real Time, const char * Name = "array.csv");

void WriteVector(const std::vector<Real>* Vector, const char * Name = "vector.csv");

void Plot(sf::Uint32 GridSize, sf::Uint8* Pixels, const Real * omega);

#endif