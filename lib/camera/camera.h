#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>
#include <cuda_runtime.h>

__device__ __host__ class Camera
{
public:
    glm::vec3 position;
    glm::vec3 target;
    glm::vec3 up;

    float rotationSpeed;

    __device__ __host__ Camera(glm::vec3 position, glm::vec3 target, glm::vec3 up, float rotationSpeed);

    __device__ __host__ void rotate(float x, float y);

    __device__ __host__ void move(float z);
};

#endif