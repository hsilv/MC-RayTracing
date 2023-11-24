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
            return Color(0, 23, 29);
        }
        return Color(173, 216, 230);
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

    diffuseLight = diffuseColor * light->intensity * diffuseLightIntensity * mat.albedo;
    specularLight = light->color * light->intensity * specularLightIntensity * mat.specularAlbedo;
    color = color + diffuseLight + specularLight;

    if (inShadow)
    {
        color = color * 0.2f;
    }

    Color reflectedColor(0, 0, 0);
    Color refractedColor(0, 0, 0);

    if (depth < MAX_DEPTH && mat.reflectivity > 0)
    {
        glm::vec3 reflectedDirection = glm::reflect(direction, globalIntersect.normal);
        // Añadir un pequeño bias a la posición desde la que se traza el rayo reflejado
        glm::vec3 bias = 0.0001f * globalIntersect.normal;
        reflectedColor = castRay(globalIntersect.point + bias, reflectedDirection, objects, numObjects, light, depth + 1);
    }

    if (depth < MAX_DEPTH && mat.transparency > 0)
    {
        glm::vec3 refractedDirection;
        float eta = mat.refractiveIndex;
        if (glm::dot(direction, globalIntersect.normal) > 0)
        {
            // Estamos dentro del objeto, invertir la normal y el índice de refracción
            refractedDirection = glm::refract(direction, -globalIntersect.normal, eta);
        }
        else
        {
            refractedDirection = glm::refract(direction, globalIntersect.normal, 1.0f / eta);
        }
        // Añadir un pequeño bias a la posición desde la que se traza el rayo refractado
        glm::vec3 bias = 0.0001f * globalIntersect.normal;
        refractedColor = castRay(globalIntersect.point + bias, refractedDirection, objects, numObjects, light, depth + 1);
    }

    color = color * (1 - mat.reflectivity - mat.transparency) + reflectedColor * mat.reflectivity + refractedColor * mat.transparency;

    return color;
}

#endif