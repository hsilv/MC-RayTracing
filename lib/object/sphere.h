#ifndef SPHERE_H
#define SPHERE_H

#include "object.h"
#include <glm/glm.hpp>
#include "material.h"

class Sphere : public Object
{
public:
        Sphere(const glm::vec3 &center, float radius, Material material) : center(center), radius(radius), Object(material) {}

        __device__ bool rayIntersect(const glm::vec3 &origin, const glm::vec3 &direction)
        {
            glm::vec3 oc = origin - center;

            float a = glm::dot(direction, direction);
            float b = 2.0f * glm::dot(oc, direction);
            float c = glm::dot(oc, oc) - (radius * radius);

            float discriminant = b * b - 4 * a * c;

            return discriminant > 0;
        };

    private:
        glm::vec3 center;
        float radius;
    };

    #endif
