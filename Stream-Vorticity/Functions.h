#pragma once
#include <SFML/Graphics.hpp>
#include "Macros.h"

enum EWall
{
	EWall_Top,
	EWall_Bottom,
	EWall_Left,
	EWall_Right
};

void WriteArray(sf::Uint32 GridSize, const float * ArayToDebug);

void DebugArray(sf::Uint32 GridSize, const float * ArayToDebug, sf::Uint8* Pixels);

void SetVelocityColor(sf::Uint32 GridSize, const float* u, const float* v, float& HighestSqNorm, sf::Uint8* Pixels);

void DrawVelocities(sf::Uint32 GridSize, float SpriteScale, const float* u, const float* v, sf::RenderTexture& Canvas);

void SetBoundary(sf::Uint32 GridSize, float Value, EWall Wall, float * InArray);

void InitializeVorticity(sf::Uint32 GridSize, float dx, float dy, const float* u, const float* v, float* omega);

void ProduceVorticity(sf::Uint32 GridSize, float dx, float dy, const float* u, const float* v, float* omega);

void IntegrateVorticity(sf::Uint32 GridSize, float ReynoldsNumber, float dx, float dy, float dt, const float* u, const float* v, float* omega, float* helper_omega);

void IntegrateStream(sf::Uint32 GridSize, float dx, float dy, float LidVelocity, const float* omega, float * psi, float * helper);

void CalculateVelocity(sf::Uint32 GridSize, float dx, float dy, float LidVelocity, const float * psi, float * u, float * v);

void PresureStabilization(sf::Uint32 GridSize, float rho, float dx, float dy, float dt, const float * u, const float * v, float * p, float * helper);

void UpdateVelocity(sf::Uint32 GridSize, float dx, float dy, float dt, float LidVelocity, float rho, float nu, const float * p, float * u, float * v, float * un, float * vn);