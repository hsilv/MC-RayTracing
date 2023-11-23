#ifndef CUBE_H
#define CUBE_H

#include "object.h"
#include <glm/glm.hpp>
#include "material.h"
#include "intersect.h"

class Cube : public Object
{
public:
    Cube(const glm::vec3 &center, const glm::vec3 &dimensions, Material material)
        : center(center), dimensions(dimensions), Object(material)
    {
        min = center - dimensions / 2.0f;
        max = center + dimensions / 2.0f;
    }

    __device__ Intersect rayIntersect(const glm::vec3 &origin, const glm::vec3 &direction)
    {
        glm::vec3 tMin = (min - origin) / direction;
        glm::vec3 tMax = (max - origin) / direction;

        glm::vec3 realMin = glm::min(tMin, tMax);
        glm::vec3 realMax = glm::max(tMin, tMax);

        float t0 = glm::max(realMin.x, glm::max(realMin.y, realMin.z));
        float t1 = glm::min(realMax.x, glm::min(realMax.y, realMax.z));

        if (t0 > t1 || t1 < 0)
        {
            return Intersect{false};
        }

        glm::vec3 point = origin + t0 * direction;
        glm::vec3 normal;
        glm::vec2 texCoords;

        if (t0 == realMin.x)
        {
            normal = glm::vec3(1, 0, 0);
            texCoords = glm::vec2((point.y - min.y) / dimensions.y, (point.z - min.z) / dimensions.z);
        }
        else if (t0 == realMin.y)
        {
            normal = glm::vec3(0, 1, 0);
            texCoords = glm::vec2((point.x - min.x) / dimensions.x, (point.z - min.z) / dimensions.z);
        }
        else if (t0 == realMin.z)
        {
            normal = glm::vec3(0, 0, 1);
            texCoords = glm::vec2((point.x - min.x) / dimensions.x, (point.y - min.y) / dimensions.y);
        }
        else if (t0 == realMax.x)
        {
            normal = glm::vec3(-1, 0, 0);
            texCoords = glm::vec2((point.y - min.y) / dimensions.y, (point.z - min.z) / dimensions.z);
        }
        else if (t0 == realMax.y)
        {
            normal = glm::vec3(0, -1, 0);
            texCoords = glm::vec2((point.x - min.x) / dimensions.x, (point.z - min.z) / dimensions.z);
        }
        else if (t0 == realMax.z)
        {
            normal = glm::vec3(0, 0, -1);
            texCoords = glm::vec2((point.x - min.x) / dimensions.x, (point.y - min.y) / dimensions.y);
        }

        return Intersect{true, t0, point, normal, texCoords};
    };

private:
    glm::vec3 center;
    glm::vec3 dimensions;
    glm::vec3 min;
    glm::vec3 max;
};

#endif