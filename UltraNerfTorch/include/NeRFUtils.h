#ifndef NERF_UTILS_H
#define NERF_UTILS_H

#include <filesystem>
#include <opencv2/opencv.hpp>
#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#define slots Q_SLOTS

struct Point
{
    float x;
    float y;
    float z;
    Point(float x, float y, float z) : x(x), y(y), z(z) {};
};
enum class BLINE_ORIGIN
{
    BOTTOM,
    TOP,
    LEFT,
    RIGHT
};
class NeRFUtils
{
public:
    static torch::Device get_device();
    static torch::Tensor create_gaussian_kernel(int size, float mean, float std);
    static torch::Tensor raw2attenuation(torch::Tensor raw, torch::Tensor dists);
    static torch::Tensor sin_fn(const torch::Tensor &x);
    static torch::Tensor cos_fn(const torch::Tensor &x);
    static void accumulate_rays(torch::Dict<std::string, torch::Tensor> &render_results, torch::Dict<std::string, torch::Tensor> batch_render_results);
    static torch::Tensor generate_rays(Point upper_point, Point lower_point, int num_rays, BLINE_ORIGIN origin);
    static cv::Mat tensor_to_grayscale_opencv(const torch::Tensor &tensor);
};
#endif