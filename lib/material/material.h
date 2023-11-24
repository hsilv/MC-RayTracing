#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"
#include "SDL.h"

struct Texture{
    Color* colors;
    int width;
    int height;
};

struct Material {
    Color diffuse;
    float albedo;
    float specularAlbedo;
    float specularCoefficient;
    bool hasText;
    Texture texture;
    float reflectivity = 0.0f;
    float transparency = 0.0f;
    float refractiveIndex = 0.0f;
};

#endif