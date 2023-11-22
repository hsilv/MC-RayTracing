#include "SDL.h"
#include "color.h"
#include "fps.h"
#include <cuda_runtime.h>
#include "SM.h"

Color Background = {0, 0, 0};

__global__ void printHelloWorld()
{
  int index = threadIdx.x + 1;
  printf("Hello World %d\n", index);
}

int main(int argc, char *argv[])
{
  int deviceCount;
  int numCores;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount > 0)
  {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    numCores = std::min(deviceProp.multiProcessorCount * _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor), 1024);
  }
  printHelloWorld<<<1, numCores>>>();
  cudaDeviceSynchronize();

  /*-------------------------SDL CONFIGURATION----------------------------*/

  SDL_Init(SDL_INIT_VIDEO);

  SDL_Window *window = SDL_CreateWindow(
      "SDL2Test",
      SDL_WINDOWPOS_UNDEFINED,
      SDL_WINDOWPOS_UNDEFINED,
      640,
      480,
      0);

  SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
  /*----------------------------------------------------------------------*/

  bool running = true;

  while (running)
  {
    startFPS();

    SDL_Event event;

    while (SDL_PollEvent(&event))
    {
      if (event.type == SDL_QUIT)
      {
        running = false;
      }
    }
    Background = Background + Color(2, 2, 2);
    SDL_SetRenderDrawColor(renderer, Background.getRed(), Background.getGreen(), Background.getBlue(), SDL_ALPHA_OPAQUE);
    SDL_RenderClear(renderer);
    SDL_RenderPresent(renderer);
    SDL_Delay(14);
    endFPS(window);
  }

  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
}
