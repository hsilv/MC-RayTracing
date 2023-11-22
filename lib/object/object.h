#ifndef OBJECT_H
#define OBJECT_H

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <iostream>

/* class Sphere; */

class Object
{
public:
    __device__ bool rayIntersect(const glm::vec3 &origin, const glm::vec3 &direction);
};

enum ObjectType
{
    SPHERE,
    // Otros tipos de objetos
};

class Sphere : public Object
{
public:
    Sphere(const glm::vec3 &center, float radius) : center(center), radius(radius) {}

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

struct ObjectWrapper
{
    Object *obj;
    ObjectType type;

    __device__ bool rayIntersect(const glm::vec3 &origin, const glm::vec3 &direction) const
    {
        switch (type)
        {
        case SPHERE:
            Sphere *sph = static_cast<Sphere *>(obj);
            return sph->rayIntersect(origin, direction);

        }
    }
};

#endif