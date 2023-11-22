#include "color.h"
#include "fps.h"
#include "SM.h"
#include "framebufferConfig.h"
#include <glm/glm.hpp>
#include "ray.h"
#include "object.h"
#include <vector>
#include <thrust/device_vector.h>

Color Background = {0, 0, 0};
const float ASPECT_RATIO = static_cast<float>(SCREEN_WIDTH) / static_cast<float>(SCREEN_HEIGHT);

/* std::vector<Object> objects; */
thrust::device_vector<ObjectWrapper> objects;

Sphere *dev_sphere;

__global__ void render(Point *buffer, ObjectWrapper *objects, int numObjects)
{

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = index % SCREEN_WIDTH;
  int y = index / SCREEN_WIDTH;

  float screenX = ((2.0f * x) / SCREEN_WIDTH) - 1.0f;
  float screenY = -((2.0f * y) / SCREEN_HEIGHT) + 1.0f;
  screenX *= ASPECT_RATIO;

  if (screenX > 1.0f || screenX < -1.0f)
  {
    return;
  }

  if (screenY > 1.0f || screenY < -1.0f)
  {
    return;
  }

  glm::vec3 rayDirection = glm::normalize(glm::vec3(screenX, screenY, -1.0f));
  Color pixelColor = castRay(glm::vec3(0.0f, 0.0f, 0.0f), rayDirection, objects, numObjects);
  Point p = {x, y, 0, pixelColor};

  buffer[index] = p;
}

void setUp()
{

  cudaMalloc(&dev_sphere, sizeof(Sphere));
  Sphere tempSphere = Sphere(glm::vec3(0.0f, 0.0f, -5.0f), 1.0f);
  cudaMemcpy(dev_sphere, &tempSphere, sizeof(Sphere), cudaMemcpyHostToDevice);

  ObjectWrapper sphereWrapper;

  sphereWrapper.obj = dev_sphere;
  sphereWrapper.type = ObjectType::SPHERE;

  objects.push_back(sphereWrapper);
}

void destroy()
{
   cudaFree(dev_sphere);
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
    numCores = std::min(deviceProp.multiProcessorCount * _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor), 1024);
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

  setUp();

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
    SDL_SetRenderDrawColor(renderer, Background.getRed(), Background.getGreen(), Background.getBlue(), SDL_ALPHA_OPAQUE);
    SDL_RenderClear(renderer);

    initBuffer();

    ObjectWrapper *raw_ptr = thrust::raw_pointer_cast(objects.data());

    render<<<numBlocks, numCores>>>(dev_buffer, raw_ptr, objects.size());
    cudaDeviceSynchronize();
    cudaMemcpy(host_buffer, dev_buffer, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Point), cudaMemcpyDeviceToHost);

    renderBuffer(renderer, host_buffer);

    destroyBuffer();

    endFPS(window);
  }

  destroy();

  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
}
