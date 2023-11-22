#ifndef FRAMEBUFFERCONFIG_H
#define FRAMEBUFFERCONFIG_H

#include <array>
#include <mutex>
#include "screenConfig.h"
#include <cuda_runtime.h>
#include "color.h"

struct Point
{
    int x;
    int y;
    int z;
    Color color;
};

Point *dev_buffer;
Point *host_buffer;

void initBuffer()
{
    host_buffer = new Point[SCREEN_WIDTH * SCREEN_HEIGHT];
    cudaMalloc(&dev_buffer, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Point));
}

void destroyBuffer()
{
    delete[] host_buffer;
    cudaFree(dev_buffer);
}

__device__ void drawPoint(int width, int height, Point *buffer, Point point)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (point.x < width && point.y < height && point.x >= 0 && point.y >= 0)
    {
        buffer[index] = point;
    }
}

void renderBuffer(SDL_Renderer *renderer, Point *buffer)
{
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, SCREEN_WIDTH, SCREEN_HEIGHT);

    void *texturePixels;
    int pitch;
    SDL_LockTexture(texture, NULL, &texturePixels, &pitch);
    SDL_SetTextureBlendMode(texture, SDL_BLENDMODE_BLEND);

    Uint32 format = SDL_PIXELFORMAT_ARGB8888;
    SDL_PixelFormat *mappingFormat = SDL_AllocFormat(format);

    Uint32 *texturePixels32 = static_cast<Uint32 *>(texturePixels);
    for (int i = 0; i < SCREEN_WIDTH * SCREEN_HEIGHT; i++)
    {   
        Color &color = buffer[i].color;
        int x = buffer[i].x;
        int y = SCREEN_HEIGHT - buffer[i].y - 1; // Reverse the order of rows
        int index = y * (pitch / sizeof(Uint32)) + x;
        texturePixels32[index] = SDL_MapRGBA(mappingFormat, color.getRed(), color.getGreen(), color.getBlue(), 255);
    }

    SDL_UnlockTexture(texture);
    SDL_Rect textureRect = {0, 0, SCREEN_WIDTH, SCREEN_HEIGHT};
    SDL_RenderCopy(renderer, texture, NULL, &textureRect);
    SDL_DestroyTexture(texture);

    SDL_RenderPresent(renderer);
}

#endif