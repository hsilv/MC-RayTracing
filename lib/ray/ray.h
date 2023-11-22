#ifndef RAY_H
#define RAY_H

#include "color.h"
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "object.h"
#include <vector>

__device__ Color castRay(const glm::vec3 &origin, const glm::vec3 &direction, ObjectWrapper* objects, int numObjects)
{
    for (int i = 0; i < numObjects; i++)
    {
        if (objects[i].rayIntersect(origin, direction))
        {
            Material mat = objects[i].obj->material;
            Color diffuseLight = mat.diffuse;
            Color color = diffuseLight;
            return color;
        }
    }

    return Color{173, 216, 230};
}

#endif