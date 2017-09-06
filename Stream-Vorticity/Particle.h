#pragma once
#include "Macros.h"
#include <SFML/Graphics.hpp>

using namespace sf;

class Particle
{
public:
	void SetParticle(Vector2f InTopLeft, Vector2f InBottommRight, Uint32 i, Uint32 j, Real * InU, Real * InV);

	void Update(RenderWindow &Window, Real dt);
private:
	void SetPosition(Uint32 i, Uint32 j);

public:
	Real * u; 
	Real * v;

private:
	CircleShape		Circle;
	Real			ParticleRadius;
	Vector2f		TopLeft;
	Vector2f		BottommRight;
	Vector2f		CavitySize;
	Vector2f		Position;
	Uint32			Cavity_i;
	Uint32			Cavity_j;
};