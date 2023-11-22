#include "color.h"
#include "fps.h"
#include "SM.h"
#include "framebufferConfig.h"
#include <glm/glm.hpp>
#include "ray.h"
#include "object.h"
#include <vector>
#include <thrust/device_vector.h>
#include "light.h"

Color Background = {0, 0, 0};
const float ASPECT_RATIO = static_cast<float>(SCREEN_WIDTH) / static_cast<float>(SCREEN_HEIGHT);

thrust::device_vector<Light> lights;
std::vector<Light *> lightPointers;
thrust::device_vector<ObjectWrapper> objects;
std::vector<Material *> matPointers;


__global__ void render(Point *buffer, ObjectWrapper *objects, int numObjects, Light* lights, int numLights)
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

  glm::vec3 rayDirection = glm::normalize(glm::vec3(screenX, screenY, -1.0f));
  Color pixelColor = castRay(glm::vec3(0.0f, 0.0f, 0.0f), rayDirection, objects, numObjects, lights, numLights);
  Point p = {x, y, 0, pixelColor};

  buffer[index] = p;
}

void setUp()
{

  Light *dev_light;
  cudaMalloc(&dev_light, sizeof(Light));
  Light light{glm::vec3(-20.0f, -20.0f, 20.0f), 1.5f};
  cudaMemcpy(dev_light, &light, sizeof(Light), cudaMemcpyHostToDevice);

  lights.push_back(light);
  lightPointers.push_back(dev_light);

  Material *dev_rubber;
  cudaMalloc(&dev_rubber, sizeof(Material));
  Material tempRubber = Material{Color(80, 0, 0)};
  cudaMemcpy(dev_rubber, &tempRubber, sizeof(Material), cudaMemcpyHostToDevice);

  matPointers.push_back(dev_rubber);

  Material *dev_ivory;
  cudaMalloc(&dev_ivory, sizeof(Material));
  Material tempIvory = Material{Color(100, 100, 80)};
  cudaMemcpy(dev_ivory, &tempIvory, sizeof(Material), cudaMemcpyHostToDevice);

  matPointers.push_back(dev_ivory);


  Sphere *dev_sphere;
  cudaMalloc(&dev_sphere, sizeof(Sphere));
  Sphere tempSphere = Sphere(glm::vec3(0.0f, 0.0f, -5.0f), 1.0f, tempRubber);
  cudaMemcpy(dev_sphere, &tempSphere, sizeof(Sphere), cudaMemcpyHostToDevice);

  ObjectWrapper sphereWrapper1;

  sphereWrapper1.obj = dev_sphere;
  sphereWrapper1.type = ObjectType::SPHERE;


  Sphere *dev_sphere2;
  cudaMalloc(&dev_sphere2, sizeof(Sphere));
  Sphere tempSphere2 = Sphere(glm::vec3(-1.0f, 0.0f, -3.5f), 1.0f, tempIvory);
  cudaMemcpy(dev_sphere2, &tempSphere2, sizeof(Sphere), cudaMemcpyHostToDevice);

  ObjectWrapper sphereWrapper2;

  sphereWrapper2.obj = dev_sphere2;
  sphereWrapper2.type = ObjectType::SPHERE;

  objects.push_back(sphereWrapper1);
  objects.push_back(sphereWrapper2);
}

void destroy()
{

  while (objects.size() == 0)
  {
    ObjectWrapper obj = objects.back();
    cudaFree(obj.obj);
    objects.pop_back();
  }

  while(matPointers.size() == 0){
    Material *mat = matPointers.back();
    cudaFree(mat);
    matPointers.pop_back();
  }

  while(lightPointers.size() == 0){
    Light *light = lightPointers.back();
    cudaFree(light);
    lightPointers.pop_back();
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
    Light *raw_lights = thrust::raw_pointer_cast(lights.data());

    render<<<numBlocks, numCores>>>(dev_buffer, raw_ptr, objects.size(), raw_lights, lightPointers.size());
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
