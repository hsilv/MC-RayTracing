#include "camera.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <cuda_runtime.h>

__device__ __host__ Camera::Camera(glm::vec3 position, glm::vec3 target, glm::vec3 up, float rotationSpeed) : position(position), target(target), up(up), rotationSpeed(rotationSpeed) {}

__device__ __host__ void Camera::rotate(float x, float y)
{
    glm::quat quatY = glm::angleAxis(glm::radians(y * rotationSpeed), glm::vec3(0, 1, 0));
    glm::quat quatX = glm::angleAxis(glm::radians(x * rotationSpeed), glm::vec3(1, 0, 0));

    glm::vec3 newPosition = quatY * quatX * (position - target) + target;

    if (newPosition == glm::vec3(0.0f, 0.0f, 0.0f))
    {
        position = glm::vec3(0.0f, 0.0f, 0.1f);
    }
    else
    {
        position = newPosition;
    }
}

__device__ __host__ void Camera::move(float z)
{
    glm::vec3 forward = glm::normalize(target - position);
    glm::vec3 newPosition = position + forward * z;

    if (newPosition == glm::vec3(0.0f, 0.0f, 0.0f))
    {
        position = glm::vec3(0.0f, 0.0f, 0.1f);
    }
    else
    {
        position = newPosition;
    }
}
