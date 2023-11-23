#ifndef LIGHT_H
#define LIGHT_H

#include "color.h"
#include <glm/glm.hpp>

struct Light {
    glm::vec3 position;
    float intensity;
    Color color;
};

#endif