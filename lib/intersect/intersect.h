#ifndef INTERSECT_H
#define INTERSECT_H

#include <glm/glm.hpp>

struct Intersect
{
    bool intersected = false;
    float dist = 0.0f;
    glm::vec3 point;
    glm::vec3 normal;
};


#endif  