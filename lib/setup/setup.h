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

void addStair(glm::vec3 center, glm::vec3 size, int orientation, Material mat)
{
    glm::vec3 stepSize;
    glm::vec3 firstStepCenter;
    glm::vec3 secondStepCenter;
    glm::vec3 thirdStepCenter;
    if (orientation == 0)
    {
        stepSize = {size.x, size.y / 2.0f, size.z / 2.0f};
        firstStepCenter = {center.x, center.y + (stepSize.y) / 2.0f, center.z + (stepSize.z) / 2.0f};
        secondStepCenter = {center.x, center.y - (stepSize.y) / 2.0f, center.z - (stepSize.z) / 2.0f};
        thirdStepCenter = {center.x, center.y + (stepSize.y) / 2.0f, center.z - (stepSize.z) / 2.0f};
    }
    else if (orientation == 1)
    {
        stepSize = {size.x, size.y / 2.0f, size.z / 2.0f};
        firstStepCenter = {center.x, center.y - (stepSize.y) / 2.0f, center.z + (stepSize.z) / 2.0f};
        secondStepCenter = {center.x, center.y - (stepSize.y) / 2.0f, center.z - (stepSize.z) / 2.0f};
        thirdStepCenter = {center.x, center.y + (stepSize.y) / 2.0f, center.z - (stepSize.z) / 2.0f};
    }
    else if (orientation == 2)
    {
        stepSize = {size.x, size.y / 2.0f, size.z / 2.0f};
        firstStepCenter = {center.x, center.y - (stepSize.y) / 2.0f, center.z + (stepSize.z) / 2.0f};
        secondStepCenter = {center.x, center.y - (stepSize.y) / 2.0f, center.z - (stepSize.z) / 2.0f};
        thirdStepCenter = {center.x, center.y + (stepSize.y) / 2.0f, center.z + (stepSize.z) / 2.0f};
    }
    else if (orientation == 3)
    {
        stepSize = {size.x, size.y / 2.0f, size.z};
        firstStepCenter = {center.x, center.y + (stepSize.y) / 2.0f, center.z};
        secondStepCenter = {center.x - (stepSize.x) / 4.0f, center.y - (stepSize.y) / 2.0f, center.z - (stepSize.z) / 4.0f};
    }

    else if (orientation == 4)
    {
        stepSize = {size.x, size.y / 2.0f, size.z};
        firstStepCenter = {center.x, center.y + (stepSize.y) / 2.0f, center.z};
        secondStepCenter = {center.x - (stepSize.x) / 4.0f, center.y - (stepSize.y) / 2.0f, center.z + (stepSize.z) / 4.0f};
    }
    else if (orientation == 5)
    {
        stepSize = {size.x / 2.0f, size.y / 2.0f, size.z};
        firstStepCenter = {center.x - (stepSize.x) / 2.0f, center.y + (stepSize.y) / 2.0f, center.z};
        secondStepCenter = {center.x - (stepSize.x) / 2.0f, center.y - (stepSize.y) / 2.0f, center.z};
        thirdStepCenter = {center.x + (stepSize.x) / 2.0f, center.y + (stepSize.y) / 2.0f, center.z};
    }
    else if (orientation == 6)
    {
        stepSize = {size.x, size.y / 2.0f, size.z / 2.0f};
        firstStepCenter = {center.x, center.y + (stepSize.y) / 2.0f, center.z + (stepSize.z) / 2.0f};
        secondStepCenter = {center.x, center.y - (stepSize.y) / 2.0f, center.z + (stepSize.z) / 2.0f};
        thirdStepCenter = {center.x, center.y + (stepSize.y) / 2.0f, center.z - (stepSize.z) / 2.0f};
    }
    else if (orientation == 7)
    {
        stepSize = {size.x, size.y / 2.0f, size.z / 2.0f};
        firstStepCenter = {center.x, center.y + (stepSize.y) / 2.0f, center.z - (stepSize.z) / 2.0f};
        secondStepCenter = {center.x, center.y - (stepSize.y) / 2.0f, center.z + (stepSize.z) / 2.0f};
        thirdStepCenter = {center.x, center.y - (stepSize.y) / 2.0f, center.z - (stepSize.z) / 2.0f};
    }

    if (orientation == 3 || orientation == 4)
    {

        addCube(firstStepCenter, stepSize, mat);
        addCube(secondStepCenter, glm::vec3(stepSize.x / 2.0f, stepSize.y, stepSize.z / 2.0f), mat);
    }
    else
    {
        addCube(firstStepCenter, stepSize, mat);
        addCube(secondStepCenter, stepSize, mat);
        addCube(thirdStepCenter, stepSize, mat);
    }
}

void addGrassBlock(glm::vec3 center, glm::vec3 size, Material top, Material block)
{
    addCube(glm::vec3(center.x, center.y - (size.y / 2.0f), center.z), glm::vec3(size.x, size.y / 20.0f, size.z), top);
    addCube(center, size, block);
}

void setUp(SDL_Renderer *ren)
{
    initImageLoader();
    loadImage("wood", "./src/wood.bmp");
    loadImage("log", "./src/wood.jpg");
    loadImage("cobblestone", "./src/cobblestone.jpg");
    loadImage("grass", "./src/grass.jpg");
    loadImage("earthgrass", "./src/earthgrass.jpg");
    loadImage("obsidian", "./src/obsidian.jpg");
    loadImage("water", "./src/water.jpeg");
    loadImage("portal", "./src/portal.png");

    Texture portal = getTexture("portal");
    addTexture(&portal);

    Texture water = getTexture("water");
    addTexture(&water);

    Texture obsidian = getTexture("obsidian");
    addTexture(&obsidian);

    Texture earthgrass = getTexture("earthgrass");
    addTexture(&earthgrass);

    Texture log = getTexture("log");
    addTexture(&log);

    Texture wood = getTexture("wood");
    addTexture(&wood);

    Texture cobblestone = getTexture("cobblestone");
    addTexture(&cobblestone);

    Texture grass = getTexture("grass");
    addTexture(&grass);

    Light light{glm::vec3(4.5f, -1.0f, -3.5f), 0.5f, Color(255, 100, 0), 0.5f};
    addLight(light);

    Light light2{glm::vec3(-10.0f, -10.0f, -5.0f), 0.15f, Color(255, 255, 255), 0.2f};
    addLight(light2);

    Material oakWood = Material{Color(100, 80, 0), 0.7f, 0.1f, 50.0f, true, wood};
    addMaterial(oakWood);

    Material logMat = Material{Color(100, 80, 0), 0.6f, 0.4f, 50.0f, true, log};
    addMaterial(logMat);

    Material cobblestoneMat = Material{Color(100, 80, 0), 0.6f, 0.04f, 50.0f, true, cobblestone};
    addMaterial(cobblestoneMat);

    Material grassMat = Material{Color(100, 80, 0), 0.6f, 0.4f, 50.0f, true, grass};
    addMaterial(grassMat);

    Material earthgrassMat = Material{Color(100, 80, 0), 0.6f, 0.4f, 50.0f, true, earthgrass};
    addMaterial(earthgrassMat);

    Material obsidianMat = Material{Color(100, 80, 0), 0.6f, 0.4f, 50.0f, true, obsidian};
    addMaterial(obsidianMat);

    Material waterMat = Material{Color(100, 80, 0), 0.6f, 0.4f, 50.0f, true, water, 0.5f};
    addMaterial(waterMat);

    Material portalMat = Material{Color(100, 80, 0), 0.1f, 0.1f, 1.0f, true, portal, 0.3f, 0.0f, 0.0f};
    addMaterial(portalMat);

    Material glass = Material{Color(18,25,32), 0.1f, 0.1f, 1.0f, false, portal, 0.0f, 0.7f, 0.5f};
    addMaterial(glass);

    int numStairs = 4;

    glm::vec3 blockSize = {1.0f, 1.0f, 1.0f};

    glm::vec3 portalSize = {0.2f, 1.0f, 1.0f};
    for (int x = 0; x < numStairs; x++)
    {
        if (x != numStairs - 1)
        {
            addStair(glm::vec3(3.0f, -x, -x - 1.0f), blockSize, 2, oakWood);
            addStair(glm::vec3(-3.0f, -x, -x - 1.0f), blockSize, 2, oakWood);
            addStair(glm::vec3(3.0f, -x, x - 6.0f), blockSize, 7, oakWood);
            addStair(glm::vec3(-3.0f, -x, x - 6.0f), blockSize, 7, oakWood);
        }
        addStair(glm::vec3(3.0f, -x, -x), blockSize, 0, oakWood);
        addStair(glm::vec3(3.0f, -x, x - 7.0f), blockSize, 6, oakWood);
        addStair(glm::vec3(2.0f, -x, -x), blockSize, 0, oakWood);
        addStair(glm::vec3(2.0f, -x, x - 7.0f), blockSize, 6, oakWood);
        addStair(glm::vec3(1.0f, -x, -x), blockSize, 0, oakWood);
        addStair(glm::vec3(1.0f, -x, x - 7.0f), blockSize, 6, oakWood);
        addStair(glm::vec3(0.0f, -x, -x), blockSize, 0, oakWood);
        addStair(glm::vec3(0.0f, -x, x - 7.0f), blockSize, 6, oakWood);
        addStair(glm::vec3(-3.0f, -x, -x), blockSize, 0, oakWood);
        addStair(glm::vec3(-3.0f, -x, x - 7.0f), blockSize, 6, oakWood);
        addStair(glm::vec3(-2.0f, -x, -x), blockSize, 0, oakWood);
        addStair(glm::vec3(-2.0f, -x, x - 7.0f), blockSize, 6, oakWood);
        addStair(glm::vec3(-1.0f, -x, -x), blockSize, 0, oakWood);
        addStair(glm::vec3(-1.0f, -x, x - 7.0f), blockSize, 6, oakWood);
    }

    addCube(glm::vec3(0.0f, 0.0f, -1.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(1.0f, 0.0f, -1.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(2.0f, 0.0f, -1.0f), blockSize, logMat);
    addCube(glm::vec3(-1.0f, 0.0f, -1.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(-2.0f, 0.0f, -1.0f), blockSize, logMat);

    addCube(glm::vec3(1.0f, 1.0f, -1.0f), blockSize, logMat);
    addCube(glm::vec3(-1.0f, 1.0f, -1.0f), blockSize, logMat);
    addCube(glm::vec3(2.0f, 1.0f, -1.0f), blockSize, logMat);
    addCube(glm::vec3(-2.0f, 1.0f, -1.0f), blockSize, logMat);
    addCube(glm::vec3(2.0f, 2.0f, -1.0f), blockSize, logMat);
    addCube(glm::vec3(-2.0f, 2.0f, -1.0f), blockSize, logMat);
    addCube(glm::vec3(2.0f, 3.0f, -1.0f), blockSize, logMat);
    addCube(glm::vec3(-2.0f, 3.0f, -1.0f), blockSize, logMat);
    addCube(glm::vec3(-1.0f, 3.0f, -1.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(1.0f, 3.0f, -1.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(0.0f, 3.0f, -1.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(-1.0f, 2.0f, -1.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(1.0f, 2.0f, -1.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(0.0f, 2.0f, -1.0f), blockSize, cobblestoneMat);

    addCube(glm::vec3(2.0f, 3.0f, -2.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(2.0f, 2.0f, -2.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(2.0f, 1.0f, -2.0f), blockSize, logMat);
    addCube(glm::vec3(2.0f, 0.0f, -2.0f), blockSize, cobblestoneMat);

    addCube(glm::vec3(2.0f, 3.0f, -3.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(2.0f, 3.0f, -4.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(2.0f, 3.0f, -5.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(2.0f, 3.0f, -6.0f), blockSize, logMat);

    addStair(glm::vec3(3.0f, 3.0f, -2.0f), blockSize, 3, cobblestoneMat);
    addStair(glm::vec3(3.0f, 3.0f, -3.0f), blockSize, 5, cobblestoneMat);
    addStair(glm::vec3(3.0f, 3.0f, -4.0f), blockSize, 5, cobblestoneMat);
    addStair(glm::vec3(3.0f, 3.0f, -5.0f), blockSize, 4, cobblestoneMat);

    addCube(glm::vec3(2.0f, 2.0f, -5.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(2.0f, 2.0f, -6.0f), blockSize, logMat);

    addCube(glm::vec3(2.0f, 1.0f, -5.0f), blockSize, logMat);
    addCube(glm::vec3(2.0f, 1.0f, -6.0f), blockSize, logMat);

    addCube(glm::vec3(2.0f, 0.0f, -5.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(2.0f, 0.0f, -6.0f), blockSize, logMat);

    addCube(glm::vec3(2.0f, 0.0f, -4.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(2.0f, 0.0f, -3.0f), blockSize, cobblestoneMat);

    addCube(glm::vec3(2.0f, -1.0f, -4.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(2.0f, -1.0f, -3.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(2.0f, -1.0f, -2.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(2.0f, -1.0f, -5.0f), blockSize, cobblestoneMat);

    addCube(glm::vec3(2.0f, -2.0f, -4.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(2.0f, -2.0f, -3.0f), blockSize, cobblestoneMat);

    addCube(glm::vec3(-2.0f, 3.0f, -6.0f), blockSize, logMat);
    addCube(glm::vec3(-1.0f, 3.0f, -6.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(0.0f, 3.0f, -6.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(1.0f, 3.0f, -6.0f), blockSize, cobblestoneMat);

    addCube(glm::vec3(-2.0f, 2.0f, -6.0f), blockSize, logMat);
    addCube(glm::vec3(-1.0f, 2.0f, -6.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(0.0f, 2.0f, -6.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(1.0f, 2.0f, -6.0f), blockSize, cobblestoneMat);

    addCube(glm::vec3(-2.0f, 1.0f, -6.0f), blockSize, logMat);
    addCube(glm::vec3(-1.0f, 1.0f, -6.0f), blockSize, logMat);
    addCube(glm::vec3(1.0f, 1.0f, -6.0f), blockSize, logMat);

    addCube(glm::vec3(-2.0f, 0.0f, -6.0f), blockSize, logMat);
    addCube(glm::vec3(-1.0f, 0.0f, -6.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(0.0f, 0.0f, -6.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(1.0f, 0.0f, -6.0f), blockSize, cobblestoneMat);

    addCube(glm::vec3(-1.0f, 3.0f, -5.0f), blockSize, oakWood);
    addCube(glm::vec3(0.0f, 3.0f, -5.0f), blockSize, oakWood);
    addCube(glm::vec3(1.0f, 3.0f, -5.0f), blockSize, oakWood);

    addCube(glm::vec3(-1.0f, 3.0f, -4.0f), blockSize, oakWood);
    addCube(glm::vec3(0.0f, 3.0f, -4.0f), blockSize, oakWood);
    addCube(glm::vec3(1.0f, 3.0f, -4.0f), blockSize, oakWood);

    addCube(glm::vec3(-1.0f, 3.0f, -3.0f), blockSize, oakWood);
    addCube(glm::vec3(0.0f, 3.0f, -3.0f), blockSize, oakWood);
    addCube(glm::vec3(1.0f, 3.0f, -3.0f), blockSize, oakWood);

    addCube(glm::vec3(-1.0f, 3.0f, -2.0f), blockSize, oakWood);
    addCube(glm::vec3(0.0f, 3.0f, -2.0f), blockSize, oakWood);
    addCube(glm::vec3(1.0f, 3.0f, -2.0f), blockSize, oakWood);

    addCube(glm::vec3(-2.0f, 3.0f, -4.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(-2.0f, 3.0f, -3.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(-2.0f, 3.0f, -2.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(-2.0f, 3.0f, -5.0f), blockSize, cobblestoneMat);

    addCube(glm::vec3(-2.0f, 2.0f, -4.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(-2.0f, 2.0f, -3.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(-2.0f, 2.0f, -2.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(-2.0f, 2.0f, -5.0f), blockSize, cobblestoneMat);

    addCube(glm::vec3(-2.0f, 1.0f, -5.0f), blockSize, logMat);
    addCube(glm::vec3(-2.0f, 1.0f, -2.0f), blockSize, logMat);

    addCube(glm::vec3(-2.0f, 0.0f, -4.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(-2.0f, 0.0f, -3.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(-2.0f, 0.0f, -2.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(-2.0f, 0.0f, -5.0f), blockSize, cobblestoneMat);

    addCube(glm::vec3(-2.0f, -1.0f, -4.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(-2.0f, -1.0f, -3.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(-2.0f, -1.0f, -2.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(-2.0f, -1.0f, -5.0f), blockSize, cobblestoneMat);

    addCube(glm::vec3(-2.0f, -2.0f, -3.0f), blockSize, cobblestoneMat);
    addCube(glm::vec3(-2.0f, -2.0f, -4.0f), blockSize, cobblestoneMat);

    int blockX = 6;
    int blockZ = 14;

    addGrassBlock(glm::vec3(0.0f, 4.0f, 0.0f), blockSize, grassMat, earthgrassMat);
    for (float x = 0.0f; x < blockX; x++)
    {
        for (float z = 0.0f; z < blockZ; z++)
        {
            if ((-z > -2.0f || -z < -5.0f) || (x > 2.0f))
            {
                addGrassBlock(glm::vec3(x, 4.0f, -z), blockSize, grassMat, earthgrassMat);
                addGrassBlock(glm::vec3(-x, 4.0f, -z), blockSize, grassMat, earthgrassMat);
            }
        }
    }

    addCube(glm::vec3(-3.0f, 3.0f, -9.0f), blockSize, obsidianMat);
    addCube(glm::vec3(-3.0f, 3.0f, -10.0f), blockSize, obsidianMat);
    addCube(glm::vec3(-3.0f, 3.0f, -11.0f), blockSize, obsidianMat);
    addCube(glm::vec3(-3.0f, 3.0f, -12.0f), blockSize, obsidianMat);

    addCube(glm::vec3(-3.0f, -1.0f, -9.0f), blockSize, obsidianMat);
    addCube(glm::vec3(-3.0f, -1.0f, -10.0f), blockSize, obsidianMat);
    addCube(glm::vec3(-3.0f, -1.0f, -11.0f), blockSize, obsidianMat);
    addCube(glm::vec3(-3.0f, -1.0f, -12.0f), blockSize, obsidianMat);

    addCube(glm::vec3(-3.0f, 0.0f, -9.0f), blockSize, obsidianMat);
    addCube(glm::vec3(-3.0f, 1.0f, -9.0f), blockSize, obsidianMat);
    addCube(glm::vec3(-3.0f, 2.0f, -9.0f), blockSize, obsidianMat);
    addCube(glm::vec3(-3.0f, 3.0f, -9.0f), blockSize, obsidianMat);

    addCube(glm::vec3(-3.0f, 0.0f, -12.0f), blockSize, obsidianMat);
    addCube(glm::vec3(-3.0f, 1.0f, -12.0f), blockSize, obsidianMat);
    addCube(glm::vec3(-3.0f, 2.0f, -12.0f), blockSize, obsidianMat);
    addCube(glm::vec3(-3.0f, 3.0f, -12.0f), blockSize, obsidianMat);

    addCube(glm::vec3(-3.0f, 2.0f, -11.0f), portalSize, portalMat);
    addCube(glm::vec3(-3.0f, 2.0f, -10.0f), portalSize, portalMat);
    addCube(glm::vec3(-3.0f, 1.0f, -11.0f), portalSize, portalMat);
    addCube(glm::vec3(-3.0f, 1.0f, -10.0f), portalSize, portalMat);
    addCube(glm::vec3(-3.0f, 0.0f, -11.0f), portalSize, portalMat);
    addCube(glm::vec3(-3.0f, 0.0f, -10.0f), portalSize, portalMat);


/*     addSphere(glm::vec3(6.0f, 2.0f, -7.0f), 1.0f, glass); */
    addCube(glm::vec3(0.0f, 1.0f, -6.1f), glm::vec3(1.0f, 1.0f, 1.0f), glass);

     addCube(glm::vec3(0.0f, 1.0f, -0.9f), glm::vec3(1.0f, 1.0f, 1.0f), glass);


     addCube(glm::vec3(-2.1f, 1.0f, -4.0f), glm::vec3(1.0f, 1.0f, 1.0f), glass);

     addCube(glm::vec3(-2.1f, 1.0f, -3.0f), glm::vec3(1.0f, 1.0f, 1.0f), glass);
}

#endif