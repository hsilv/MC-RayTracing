#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include "fps.h"
#include "SM.h"
#include "framebufferConfig.h"
#include <glm/glm.hpp>
#include "ray.h"
#include "object.h"
#include <vector>
#include "camera.h"
#include "random.h"
#include "setup.h"

#define cudaCheckError()                                                               \
  {                                                                                    \
    cudaError_t e = cudaGetLastError();                                                \
    if (e != cudaSuccess)                                                              \
    {                                                                                  \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                              \
    }                                                                                  \
  }

Color Background = {0, 0, 0};
const float ASPECT_RATIO = static_cast<float>(SCREEN_WIDTH) / static_cast<float>(SCREEN_HEIGHT);

Camera camera(glm::vec3(0.0f, 0.0f, 2.0f), glm::vec3(0.0f, 0.0f, -7.0f), glm::vec3(0.0f, 1.0f, 0.0f), 10.0f);

__global__ void render(Point *buffer, ObjectWrapper *objects, int numObjects, Light *lights, int numLights, Camera camera)
{
  float fov = FOV;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = index % SCREEN_WIDTH;
  int y = index / SCREEN_WIDTH;

  float screenX = ((2.0f * (x + 0.5f)) / SCREEN_WIDTH) - 1.0f;
  float screenY = -((2.0f * (y + 0.5f)) / SCREEN_HEIGHT) + 1.0f;
  screenX *= ASPECT_RATIO;
  screenX *= tan(fov / 2.0f);
  screenY *= tan(fov / 2.0f);

  glm::vec3 cameraDirection = glm::normalize(camera.target - camera.position);

  glm::vec3 cameraX = glm::normalize(glm::cross(cameraDirection, camera.up));
  glm::vec3 cameraY = glm::normalize(glm::cross(cameraX, cameraDirection));
  glm::vec3 rayDirection = glm::normalize(cameraX * screenX + cameraY * screenY + cameraDirection);
  Color pixelColor = {1, 1, 1};
  float factor = 1.0f / numLights;
  pixelColor = pixelColor + castRay(camera.position, rayDirection, objects, numObjects, &lights[0]);
  pixelColor = pixelColor * factor;
  Point p = {x, y, 0, pixelColor};

  buffer[index].x = p.x;
  buffer[index].y = p.y;
  buffer[index].z = p.z;
  buffer[index].color = buffer[index].color + p.color;
}

void destroy()
{

  while (objects.size() == 0)
  {
    ObjectWrapper obj = objects.back();
    cudaFree(obj.obj);
    objects.pop_back();
  }

  while (matPointers.size() == 0)
  {
    Material *mat = matPointers.back();
    cudaFree(mat);
    matPointers.pop_back();
  }

  while (lightPointers.size() == 0)
  {
    Light *light = lightPointers.back();
    cudaFree(light);
    lightPointers.pop_back();
  }

  while (texturePointers.size() == 0)
  {
    Texture *tex = texturePointers.back();
    cudaFree(tex);
    texturePointers.pop_back();
  }
}

int main(int argc, char *argv[])
{

  /*-------------------------CUDA CONFIGURATION----------------------------*/

  int deviceCount;
  int numCores;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount > 0)
  {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    numCores = std::min(deviceProp.multiProcessorCount * _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor), 512);
  }

  int numBlocks = (SCREEN_WIDTH * SCREEN_HEIGHT + numCores - 1) / numCores;

  /*-------------------------SDL CONFIGURATION----------------------------*/

  SDL_Init(SDL_INIT_VIDEO);

  SDL_Window *window = SDL_CreateWindow(
      "SDL2Test",
      SDL_WINDOWPOS_UNDEFINED,
      SDL_WINDOWPOS_UNDEFINED,
      SCREEN_WIDTH,
      SCREEN_HEIGHT,
      0);

  SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
  /*----------------------------------------------------------------------*/

  bool running = true;

  setUp(renderer);

  random_init<<<numBlocks, numCores>>>(1550);
  cudaDeviceSynchronize();

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

      if (event.type == SDL_KEYDOWN)
      {
        switch (event.key.keysym.sym)
        {
        case SDLK_UP:
          if (camera.position.z > 0.0f)
          {
            camera.move(0.08f * 2.0f);
          }
          else
          {
            camera.move(-0.08f * 2.0f);
          }
          break;
        case SDLK_DOWN:
          if (camera.position.z > 0.0f)
          {
            camera.move(-0.08f * 2.0f);
          }
          else
          {
            camera.move(0.08f * 2.0f);
          }
          break;

        case SDLK_w:
          camera.rotate(0.12f, 0.0f);
          break;

        case SDLK_s:
          camera.rotate(-0.12f, 0.0f);
          break;

        case SDLK_a:
          camera.rotate(0.0f, 0.12f);
          break;

        case SDLK_d:
          camera.rotate(0.0f, -0.12f);
          break;
        }
      }
    }
    SDL_SetRenderDrawColor(renderer, Background.getRed(), Background.getGreen(), Background.getBlue(), SDL_ALPHA_OPAQUE);
    SDL_RenderClear(renderer);

    initBuffer();

    ObjectWrapper *raw_ptr = thrust::raw_pointer_cast(objects.data());

    // Inicializa el buffer de puntos en la GPU
    Point *device_points;
    cudaMalloc(&device_points, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Point));
    cudaMemset(device_points, 0, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Point));

    cudaCheckError();

    // Lanza un kernel por cada luz
    for (int i = 0; i < lightPointers.size(); i++)
    {
      Light *light = thrust::raw_pointer_cast(&lights[i]);
      render<<<numBlocks, numCores>>>(device_points, raw_ptr, objects.size(), light, 1, camera);
      cudaDeviceSynchronize();
    }

    cudaCheckError();

    // Copia los puntos calculados al buffer de puntos del host
    cudaMemcpy(host_buffer, device_points, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Point), cudaMemcpyDeviceToHost);

    renderBuffer(renderer, host_buffer);

    // Libera el buffer de puntos
    cudaFree(device_points);

    destroyBuffer();

    endFPS(window);
  }

  destroy();

  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
}
