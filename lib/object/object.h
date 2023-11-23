#ifndef OBJECT_H
#define OBJECT_H

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include "material.h"
#include "intersect.h"

class Object
{
public:
    Object(const Material &material) : material(material) {}
    __device__ Intersect rayIntersect(const glm::vec3 &origin, const glm::vec3 &direction);
    Material material;
};

enum ObjectType
{
    SPHERE,
    CUBE,
    // Otros tipos de objetos
};

// ASEGURARSE QUE LAS FIGURAS SE IMPORTEN LUEGO DE LA DECLARACIÓN DE OBJECT.
#include "sphere.h"
#include "cube.h"

struct ObjectWrapper
{
    Object *obj;
    ObjectType type;

    __device__ Intersect rayIntersect(const glm::vec3 &origin, const glm::vec3 &direction) const
    {
        Sphere *sph = nullptr;
        Cube *cub = nullptr;

        switch (type)
        {
        case SPHERE:
            sph = static_cast<Sphere *>(obj);
            return sph->rayIntersect(origin, direction);
        case CUBE:
            cub = static_cast<Cube *>(obj);
            return cub->rayIntersect(origin, direction);
        }
    }
};

#endif