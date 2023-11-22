#ifndef OBJECT_H
#define OBJECT_H

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include "material.h"

/* class Sphere; */

class Object
{
public:
    Object(const Material& material) : material(material) {}
    __device__ bool rayIntersect(const glm::vec3 &origin, const glm::vec3 &direction);
    Material material;
};

enum ObjectType
{
    SPHERE,
    // Otros tipos de objetos
};

//ASEGURARSE QUE LAS FIGURAS SE IMPORTEN LUEGO DE LA DECLARACIÓN DE OBJECT.
#include "sphere.h"

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