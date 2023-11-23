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
std::vector<Color *> colorPointers;

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

/* void addColors(Color *colors){
    Color *dev_colors;
    cudaMalloc(&dev_colors, sizeof(Color) * 100);
    cudaMemcpy(dev_colors, colors, sizeof(Color) * 100, cudaMemcpyHostToDevice);

    colorPointers.push_back(dev_colors);
} */

void colorsToBMP(Color *colors, int width, int height)
{
    SDL_Surface *surface = SDL_CreateRGBSurface(0, width, height, 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0);

    if (surface == NULL)
    {
        printf("Could not create surface: %s\n", SDL_GetError());
        return;
    }

    SDL_LockSurface(surface);
    Uint32 *pixels = (Uint32 *)surface->pixels;

    for (int i = 0; i < width * height; i++)
    {
        pixels[i] = SDL_MapRGB(surface->format, colors[i].getRed(), colors[i].getGreen(), colors[i].getBlue());
    }

    SDL_UnlockSurface(surface);

    if (SDL_SaveBMP(surface, "colors.bmp") != 0)
    {
        printf("Could not save BMP: %s\n", SDL_GetError());
    }

    SDL_FreeSurface(surface);
}

void addTexture(Texture *texture)
{
    // Allocate memory for the colors array on the GPU
    Color *dev_colors;
    cudaMalloc(&dev_colors, texture->width * texture->height * sizeof(Color));
    cudaMemcpy(dev_colors, texture->colors, texture->width * texture->height * sizeof(Color), cudaMemcpyHostToDevice);

    // Update the colors pointer of the texture on the host to point to the new array
    texture->colors = dev_colors;

    // Allocate memory for the texture on the GPU
    Texture *dev_texture;
    cudaMalloc(&dev_texture, sizeof(Texture));
    cudaMemcpy(dev_texture, texture, sizeof(Texture), cudaMemcpyHostToDevice);

    texturePointers.push_back(dev_texture);
};

void setUp(SDL_Renderer *ren)
{
    initImageLoader();
    loadImage("wood", "./src/wood.bmp");
    Texture wood = getTexture("wood");
    addTexture(&wood);

    Light light{glm::vec3(5.0f, -5.0f, 10.0f), 1.5f, Color(255, 255, 255)};
    addLight(light);

    Material tempRubber = Material{Color(100, 100, 80), 0.9f, 0.1f, 10.0f, false};
    addMaterial(tempRubber);

    Material tempIvory = Material{Color(80, 0, 0), 0.6f, 0.4f, 50.0f, false};
    addMaterial(tempIvory);

    Material oakWood = Material{Color(100, 80, 0), 0.6f, 0.4f, 50.0f, true, wood};
    addMaterial(oakWood);

    addSphere(glm::vec3(1.0f, 0.0f, -5.0f), 1.0f, oakWood);

    addCube(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.5f, 0.5f), oakWood);
    addCube(glm::vec3(0.0f, 0.5f, 0.0f), glm::vec3(1.0f, 0.5f, 0.5f), oakWood);
    addCube(glm::vec3(0.0f, 0.0f, 0.5f), glm::vec3(1.0f, 0.5f, 0.5f), oakWood);
}

#endif