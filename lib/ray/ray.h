#ifndef RAY_H
#define RAY_H

#include "color.h"
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "object.h"
#include <vector>

__device__ Color castRay(const glm::vec3 &origin, const glm::vec3 &direction, ObjectWrapper *objects, int numObjects)
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

    if(!globalIntersect.intersected){
        return Color(173, 216, 230);
    }
    
    Material mat = hitObject->material;
    Color diffuseLight = mat.diffuse;
    Color color = diffuseLight;
    return color;
}

#endif