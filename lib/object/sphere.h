#ifndef SPHERE_H
#define SPHERE_H

#include "object.h"
#include <glm/glm.hpp>
#include "material.h"
#include "intersect.h"

class Sphere : public Object
{
public:
    Sphere(const glm::vec3 &center, float radius, Material material) : center(center), radius(radius), Object(material) {}

    __device__ Intersect rayIntersect(const glm::vec3 &origin, const glm::vec3 &direction)
    {
        glm::vec3 oc = origin - center;

        float a = glm::dot(direction, direction);
        float b = 2.0f * glm::dot(oc, direction);
        float c = glm::dot(oc, oc) - (radius * radius);

        float discriminant = b * b - 4 * a * c;

        if(discriminant < 0)
        {
            return Intersect{false};
        }

        float dist = (-b - sqrt(discriminant)) / (2.0f * a);

        if (dist < 0.0f)
        {
            return Intersect{false};
        }

        glm::vec3 point = origin + dist * direction;
        glm::vec3 normal = glm::normalize(point - center);

        // Calculate texture coordinates
        glm::vec2 texCoords;
        texCoords.x = 0.5f + atan2(normal.z, normal.x) / (2.0f * M_PI);
        texCoords.y = 0.5f - asin(normal.y) / M_PI;

        return Intersect{true, dist, point, normal, texCoords};
    };

private:
    glm::vec3 center;
    float radius;
};

#endif