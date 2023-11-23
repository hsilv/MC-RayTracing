#ifndef TEXT_H
#define TEXT_H

#include <iostream>
#include "SDL.h"
#include "SDL_image.h"
#include <map>
#include <string>
#include <cuda_runtime.h>

std::map<std::string, SDL_Surface *> imageSurfaces;

void initImageLoader()
{
    int imgFlags = IMG_INIT_PNG; // or IMG_INIT_JPG, depending on your needs
    if (!(IMG_Init(imgFlags) & imgFlags))
    {
        throw std::runtime_error("SDL_image could not initialize! SDL_image Error: " + std::string(IMG_GetError()));
    }
}

void loadImage(const std::string &key, const char *path)
{
    SDL_Surface *newSurface = IMG_Load(path);
    if (!newSurface)
    {
        throw std::runtime_error("Unable to load image! SDL_image Error: " + std::string(IMG_GetError()));
    }
    imageSurfaces[key] = newSurface;
}

Color getPixelColor(const std::string &key, int x, int y)
{
    auto it = imageSurfaces.find(key);
    if (it == imageSurfaces.end())
    {
        throw std::runtime_error("Image key not found!");
    }

    SDL_Surface *targetSurface = it->second;
    int bpp = targetSurface->format->BytesPerPixel;
    Uint8 *p = (Uint8 *)targetSurface->pixels + y * targetSurface->pitch + x * bpp;

    Uint32 pixelColor;
    switch (bpp)
    {
    case 1:
        pixelColor = *p;
        break;
    case 2:
        pixelColor = *(Uint16 *)p;
        break;
    case 3:
        if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
        {
            pixelColor = p[0] << 16 | p[1] << 8 | p[2];
        }
        else
        {
            pixelColor = p[0] | p[1] << 8 | p[2] << 16;
        }
        break;
    case 4:
        pixelColor = *(Uint32 *)p;
        break;
    default:
        throw std::runtime_error("Unknown format!");
    }

    SDL_Color color;
    SDL_GetRGBA(pixelColor, targetSurface->format, &color.r, &color.g, &color.b, &color.a);
    return Color{color.r, color.g, color.b, color.a};
}

Texture getTexture(const std::string& key) {
    auto it = imageSurfaces.find(key);
    if (it == imageSurfaces.end()) {
        throw std::runtime_error("Image key not found!");
    }

    SDL_Surface* targetSurface = it->second;
    int width = targetSurface->w;
    int height = targetSurface->h;
    Color* colors = new Color[width * height];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            colors[y * width + x] = getPixelColor(key, x, y) * 0.7f;
        }
    }

    return Texture{colors, width, height};
}

#endif
