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
        float tnear = -INFINITY;
        float tfar = INFINITY;
        glm::vec3 normal;

        for (int i = 0; i < 3; ++i)
        {
            if (direction[i] == 0.0f)
            {
                if (origin[i] < min[i] || origin[i] > max[i])
                {
                    return Intersect{false, 0, glm::vec3(0), glm::vec3(0), glm::vec2(0)};
                }
            }
            else
            {
                float t1 = (min[i] - origin[i]) / direction[i];
                float t2 = (max[i] - origin[i]) / direction[i];

                if (t1 > t2)
                {
                    float temp = t1;
                    t1 = t2;
                    t2 = temp;
                }

                if (t1 > tnear)
                {
                    tnear = t1;
                    normal = glm::vec3(0);
                    normal[i] = (t1 == (min[i] - origin[i]) / direction[i]) ? -1 : 1;
                }

                if (t2 < tfar)
                {
                    tfar = t2;
                }

                if (tnear > tfar || tfar < 0)
                {
                    return Intersect{false, 0, glm::vec3(0), glm::vec3(0), glm::vec2(0)};
                }
            }
        }

        glm::vec3 point = origin + tnear * direction;
        glm::vec3 localPoint = point - min;
        glm::vec2 texCoords;

        if (normal.x != 0)
        {
            texCoords.x = localPoint.z / dimensions.z;
            texCoords.y = localPoint.y / dimensions.y;
        }
        else if (normal.y != 0)
        {
            texCoords.x = localPoint.x / dimensions.x;
            texCoords.y = localPoint.z / dimensions.z;
        }
        else if (normal.z != 0)
        {
            texCoords.x = localPoint.x / dimensions.x;
            texCoords.y = localPoint.y / dimensions.y;
        }

        if (normal.x < 0 || normal.y < 0 || normal.z < 0)
        {
            texCoords = glm::vec2(1.0f) - texCoords;
        }

        return Intersect{true, tnear, point, normal, texCoords};
    }

private:
    glm::vec3 center;
    glm::vec3 dimensions;
    glm::vec3 min;
    glm::vec3 max;
};

#endif