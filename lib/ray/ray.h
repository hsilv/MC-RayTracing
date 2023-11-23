#ifndef RAY_H
#define RAY_H

#include "color.h"
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "object.h"
#include <vector>
#include "light.h"

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

__device__ Color castRay(const glm::vec3 &origin, const glm::vec3 &direction, ObjectWrapper *objects, int numObjects, Light *lights, int numLights)
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
        return Color(173, 216, 230);
    }

    Color color;

    if (numLights == 0)
    {
        return hitObject->material.diffuse;
    }

    if (numLights > 0)
    {
        Light light = lights[0];
        glm::vec3 lightDir = glm::normalize(light.position - globalIntersect.point);
        Material mat = hitObject->material;

        glm::vec3 viewDirection = glm::normalize(origin - globalIntersect.point);
        glm::vec3 reflectDirection = glm::reflect(-lightDir, globalIntersect.normal);

        float diffuseLightIntensity = glm::dot(lightDir, globalIntersect.normal);
        diffuseLightIntensity = glm::max(diffuseLightIntensity, 0.0f);

        float specularLightIntensity = 0.0f;
        if (diffuseLightIntensity > 0.0f)
        {
            float specReflection = glm::dot(viewDirection, reflectDirection);
            specularLightIntensity = pow(specReflection, mat.specularCoefficient);
        }

        // Shadow check
        bool inShadow = false;
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
        Color diffuseColor = mat.diffuse;
        if (mat.hasText)
        {
            diffuseColor = getTextureColor(globalIntersect.texCoords, mat.texture);
        }

        Color diffuseLight = diffuseColor * light.intensity * diffuseLightIntensity * mat.albedo;
        Color specularLight = light.color * light.intensity * specularLightIntensity * mat.specularAlbedo;
        color = color + diffuseLight + specularLight;

        if (inShadow)
        {
            color = color * 0.2f;
        }
    }

    return color;
}

#endif