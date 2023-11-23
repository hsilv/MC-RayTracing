#ifndef SETUP_H
#define SETUP_H

#include "light.h"
#include <vector>
#include "object.h"
#include "material.h"
#include <thrust/device_vector.h>
#include "text.h"

thrust::device_vector<Light> lights;
std::vector<Light *> lightPointers;
thrust::device_vector<ObjectWrapper> objects;
std::vector<Material *> matPointers;
std::vector<Texture *> texturePointers;

void addSphere(glm::vec3 center, float radius, Material mat)
{
    Sphere *dev_sphere;
    cudaMalloc(&dev_sphere, sizeof(Sphere));
    Sphere tempSphere = Sphere(center, radius, mat);
    cudaMemcpy(dev_sphere, &tempSphere, sizeof(Sphere), cudaMemcpyHostToDevice);

    ObjectWrapper sphereWrapper;

    sphereWrapper.obj = dev_sphere;
    sphereWrapper.type = ObjectType::SPHERE;

    objects.push_back(sphereWrapper);
}

void addCube(glm::vec3 center, glm::vec3 size, Material mat)
{
    Cube *dev_cube;
    cudaMalloc(&dev_cube, sizeof(Cube));
    Cube tempCube = Cube(center, size, mat);
    cudaMemcpy(dev_cube, &tempCube, sizeof(Cube), cudaMemcpyHostToDevice);

    ObjectWrapper cubeWrapper;

    cubeWrapper.obj = dev_cube;
    cubeWrapper.type = ObjectType::CUBE;

    objects.push_back(cubeWrapper);
}

void addMaterial(Material mat)
{
    Material *dev_mat;
    cudaMalloc(&dev_mat, sizeof(Material));
    cudaMemcpy(dev_mat, &mat, sizeof(Material), cudaMemcpyHostToDevice);

    matPointers.push_back(dev_mat);
}

void addLight(Light light)
{
    Light *dev_light;
    cudaMalloc(&dev_light, sizeof(Light));
    cudaMemcpy(dev_light, &light, sizeof(Light), cudaMemcpyHostToDevice);

    lights.push_back(light);
    lightPointers.push_back(dev_light);
}

Texture surfaceToColor(SDL_Surface *surface)
{
    int width = surface->w;
    int height = surface->h;
    Color *colors = new Color[width * height];

    SDL_LockSurface(surface);
    Uint32 *pixels = (Uint32 *)surface->pixels;

    for (int i = 0; i < width * height; i++)
    {
        Uint8 r, g, b;
        SDL_GetRGB(pixels[i], surface->format, &r, &g, &b);
        colors[i] = Color(r, g, b);
    }

    SDL_UnlockSurface(surface);

    return Texture{colors, width, height};
}

void addTexture(Texture *texture)
{
    Texture *dev_texture;
    cudaMalloc(&dev_texture, sizeof(Texture));
    cudaMemcpy(dev_texture, texture, sizeof(Texture), cudaMemcpyHostToDevice);

    texturePointers.push_back(dev_texture);
};

void setUp(SDL_Renderer *ren)
{

    SDL_Surface *Swood = loadTexture("./src/wood.jpg");
    SDL_Surface *SchWood = loadTexture("./src/choppedWood.webp");

    Texture wood = surfaceToColor(Swood);

    Light light{glm::vec3(5.0f, -5.0f, 10.0f), 1.5f, Color(255, 255, 255)};
    addLight(light);

    Material tempRubber = Material{Color(100, 100, 80), 0.9f, 0.1f, 10.0f};
    addMaterial(tempRubber);

    Material tempIvory = Material{Color(80, 0, 0), 0.6f, 0.4f, 50.0f};
    addMaterial(tempIvory);

    Material oakWood = Material{Color(100, 80, 0), 0.6f, 0.4f, 50.0f, wood};
    addMaterial(oakWood);

    addSphere(glm::vec3(1.0f, 0.0f, -5.0f), 1.0f, tempRubber);

    addCube(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.5f, 0.5f), oakWood);
    addCube(glm::vec3(0.0f, 0.5f, 0.0f), glm::vec3(1.0f, 0.5f, 0.5f), oakWood);
    addCube(glm::vec3(0.0f, 0.0f, 0.5f), glm::vec3(1.0f, 0.5f, 0.5f), oakWood);
}

#endif