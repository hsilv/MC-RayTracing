#ifndef FPS_H
#define FPS_H

#include <SDL2/SDL.h>
#include <iostream>
#include <sstream>

uint32_t frameStart, frameTick;

void startFPS()
{
    frameStart = 0;
    frameTick = 0;
    frameStart = SDL_GetTicks();
}

void endFPS(SDL_Window *window)
{
    frameTick = SDL_GetTicks() - frameStart;

    if (frameTick > 0)
    {
        std::ostringstream titleStream;
        titleStream << "FPS: " << 1000.0 / frameTick;
        SDL_SetWindowTitle(window, titleStream.str().c_str());
    }
}

#endif