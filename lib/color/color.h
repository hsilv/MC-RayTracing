#ifndef COLOR_H
#define COLOR_H
#include <iostream>
#include <cuda_runtime.h>

class Color
{
public:
    __host__ __device__ Color() : r(0), g(0), b(0) {}
    __host__ __device__ Color(uint8_t red, uint8_t green, uint8_t blue) : r(red), g(green), b(blue) {}

    __host__ __device__ Color(int red, int green, int blue, int alpha = 255)
    {
        r = uint8_t(red);
        g = uint8_t(green);
        b = uint8_t(blue);
    }

    __host__ __device__ Color(float red, float green, float blue, float alpha = 1.0f)
    {
        r = uint8_t(red * 255);
        g = uint8_t(green * 255);
        b = uint8_t(blue * 255);
    }

    __host__ __device__ uint8_t &getBlue()
    {
        return b;
    }

    __host__ __device__ const uint8_t &getBlue() const
    {
        return b;
    }

    __host__ __device__ uint8_t &getGreen()
    {
        return g;
    }

    __host__ __device__ const uint8_t &getGreen() const
    {
        return g;
    }

    __host__ __device__ uint8_t &getRed()
    {
        return r;
    }

    __host__ __device__ const uint8_t &getRed() const
    {
        return r;
    }

    __host__ __device__ Color operator+(const Color &other) const
    {
        int newR = int(r) + int(other.r);
        int newG = int(g) + int(other.g);
        int newB = int(b) + int(other.b);

        int maxColor = max(max(newR, newG), newB);

        if (maxColor > 255)
        {
            float scale = 255.0f / maxColor;
            newR = int(newR * scale);
            newG = int(newG * scale);
            newB = int(newB * scale);
        }

        return Color(newR, newG, newB);
    }
    // Overload the * operator to scale colors by a factor
    __host__ __device__ Color operator*(float factor) const
    {
        int newR = int((r / 255.0f) * factor * 255);
        int newG = int((g / 255.0f) * factor * 255);
        int newB = int((b / 255.0f) * factor * 255);

        int maxColor = max(max(newR, newG), newB);

        if (maxColor > 255)
        {
            float scale = 255.0f / maxColor;
            newR = int(newR * scale);
            newG = int(newG * scale);
            newB = int(newB * scale);
        }

        return Color(newR, newG, newB);
    }

    __host__ __device__ uint16_t toHex() const
    {
        uint16_t r_16 = (r * 31) / 255;
        uint16_t g_16 = (g * 63) / 255;
        uint16_t b_16 = (b * 31) / 255;

        return (r_16 << 11) | (g_16 << 5) | b_16;
    }

    __host__ __device__ void fromHex(uint32_t hexColor)
    {
        r = (hexColor >> 16) & 0xFF;
        g = (hexColor >> 8) & 0xFF;
        b = hexColor & 0xFF;
    }

private:
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

#endif