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

    if(numLights > 0){
        Light light = lights[0];
        glm::vec3 lightDir = glm::normalize(light.position - globalIntersect.point);
        float diffuseLightIntensity = glm::dot(lightDir, globalIntersect.normal);
        diffuseLightIntensity = glm::max(diffuseLightIntensity, 0.0f);
        Material mat = hitObject->material;
        Color diffuseLight = mat.diffuse * light.intensity * diffuseLightIntensity;
        color = diffuseLight;
    } else {
        color = hitObject->material.diffuse;
    }

    return color;
}

#endif