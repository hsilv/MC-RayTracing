#ifndef TEXT_H
#define TEXT_H

#include <iostream>
#include "SDL.h"
#include "SDL_image.h"

SDL_Texture* loadTexture(const std::string &file, SDL_Renderer *ren) {
    SDL_Texture *texture = IMG_LoadTexture(ren, file.c_str());
    if (texture == nullptr){
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "LoadTexture: %s", SDL_GetError());
    }
    return texture;
}

#endif
