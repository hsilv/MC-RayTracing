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
};

#endif