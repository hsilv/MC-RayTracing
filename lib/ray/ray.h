#ifndef RAY_H
#define RAY_H

#include "color.h"
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "object.h"
#include <vector>
#include "light.h"

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
            Intersect shadowIntersect = objects[i].rayIntersect(globalIntersect.point + 0.001f * lightDir, lightDir);
            if (shadowIntersect.intersected)
            {
                inShadow = true;
                break;
            }
        }
        Color diffuseLight = mat.diffuse * light.intensity * diffuseLightIntensity * mat.albedo;
        Color specularLight = light.color * light.intensity * specularLightIntensity * mat.specularAlbedo;
        color = diffuseLight + specularLight;

        if (inShadow)
        {
            color = color * 0.2f;
        }
    }
    else
    {
        color = hitObject->material.diffuse;
    }

    return color;
}

#endif