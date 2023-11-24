#ifndef RAY_H
#define RAY_H

#include "color.h"
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "object.h"
#include <vector>
#include "light.h"

#define MAX_DEPTH 1

__device__ Color getTextureColor(const glm::vec2 &texCoords, const Texture &texture)
{

    int x = texCoords.x * texture.width;
    int y = texCoords.y * texture.height;

    x = x % texture.width;
    y = y % texture.height;
    // Get the color from the texture
    Color color = texture.colors[y * texture.width + x];

    return color;
}

__device__ Color castRay(const glm::vec3 &origin, const glm::vec3 &direction, ObjectWrapper *objects, int numObjects, Light *light, int depth = 0)
{
    float zBuffer = INFINITY;
    Object *hitObject = nullptr;
    Intersect globalIntersect;

    for (int i = 0; i < numObjects; i++)
    {
        Intersect intersect = objects[i].rayIntersect(origin, direction);

        if (intersect.intersected && intersect.dist < zBuffer)
        {
            zBuffer = intersect.dist;
            hitObject = objects[i].obj;
            globalIntersect = intersect;
        }
    }

    if (!globalIntersect.intersected)
    {
        if (depth == MAX_DEPTH)
        {   
            return Color(18,25,32);
            /* return Color(173, 216, 230); */
            /* return Color(0+29+29, 23+29+29, 29+29+29); */
        }
        /* return Color(173, 216, 230); */
        return Color(18,25,32);
    }

    Color color;

    if (light == nullptr)
    {
        return hitObject->material.diffuse;
    }

    glm::vec3 lightDir;
    Material mat;
    glm::vec3 viewDirection;
    glm::vec3 reflectDirection;
    float diffuseLightIntensity;
    float specularLightIntensity;
    bool inShadow;
    Color diffuseColor;
    Color diffuseLight;
    Color specularLight;

    lightDir = glm::normalize(light->position - globalIntersect.point);
    mat = hitObject->material;

    viewDirection = glm::normalize(origin - globalIntersect.point);
    reflectDirection = glm::reflect(-lightDir, globalIntersect.normal);

    diffuseLightIntensity = glm::dot(lightDir, globalIntersect.normal);
    diffuseLightIntensity = glm::max(diffuseLightIntensity, 0.0f);

    specularLightIntensity = 0.0f;
    if (diffuseLightIntensity > 0.0f)
    {
        float specReflection = glm::dot(viewDirection, reflectDirection);
        specularLightIntensity = pow(specReflection, mat.specularCoefficient);
    }

    // Shadow check
    inShadow = false;
    for (int i = 0; i < numObjects; i++)
    {
        Intersect shadowIntersect = objects[i].rayIntersect(globalIntersect.point + 0.01f * lightDir, lightDir);
        if (shadowIntersect.intersected)
        {
            inShadow = true;
            break;
        }
    }

    // Texture mapping
    diffuseColor = mat.diffuse;
    if (mat.hasText)
    {
        diffuseColor = getTextureColor(globalIntersect.texCoords, mat.texture);
    }

    diffuseLight = (light->color * light->intensity * diffuseLightIntensity * light->colorIntensity) + (diffuseColor * light->intensity * diffuseLightIntensity * mat.albedo) * (1.0f / light->colorIntensity) ;
    specularLight = light->color * light->intensity * specularLightIntensity * mat.specularAlbedo;
    color = color + diffuseLight + specularLight;

    if (inShadow)
    {
        color = color * 0.2f;
    }

    Color reflectedColor(0, 0, 0);
    Color refractedColor(0, 0, 0);
    if (depth < MAX_DEPTH && mat.transparency > 0)
    {
        glm::vec3 origin = globalIntersect.point - globalIntersect.normal * 0.0001f;
        glm::vec3 refractDir = glm::refract(direction, globalIntersect.normal, mat.refractiveIndex);
        refractedColor = castRay(origin, refractDir, objects, numObjects, light, depth + 1);
    }

    if (depth < MAX_DEPTH && mat.reflectivity > 0)
    {
        glm::vec3 reflectedDirection = glm::reflect(direction, globalIntersect.normal);
        // Añadir un pequeño bias a la posición desde la que se traza el rayo reflejado
        glm::vec3 bias = 0.0001f * globalIntersect.normal;
        reflectedColor = castRay(globalIntersect.point + bias, reflectedDirection, objects, numObjects, light, depth + 1);
    }

    color = color * (1.0f - mat.reflectivity - mat.transparency) + reflectedColor * mat.reflectivity + refractedColor * mat.transparency;

    return color;
}

#endif