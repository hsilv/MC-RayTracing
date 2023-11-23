#ifndef TEXT_H
#define TEXT_H

#include <iostream>
#include "SDL.h"
#include "SDL_image.h"

SDL_Surface* loadTexture(const std::string &file) {
    SDL_Surface *texture = IMG_Load(file.c_str());
    if (texture == nullptr){
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "LoadTexture: %s", SDL_GetError());
    }
    return texture;
}

#endif
