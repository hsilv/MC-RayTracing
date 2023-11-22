#ifndef OBJECT_H
#define OBJECT_H

#include <glm/glm.hpp>
#include <cuda_runtime.h>

class Object
{
public:
   __device__ bool rayIntersect(const glm::vec3 &origin, const glm::vec3 &direction){
      return;
   };
};

#endif