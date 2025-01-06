#ifndef NERF_UTILS_H
#define NERF_UTILS_H

#include <filesystem>
#include <opencv2/opencv.hpp>
#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#define slots Q_SLOTS

// Rendering helper functions
/* void renderAndSaveOrbitViews(const NeRFRenderer &renderer, int nFrames,
                             const std::filesystem::path &outputFolder,
                             float radius = 4.0f,
                             float startDistance = 2.0f,
                             float endDistance = 5.0f, int nSamples = 64);
void saveImage(const torch::Tensor &tensor,
               const std::filesystem::path &file_path);

// Transformation and pose functions
torch::Tensor createSphericalPose(float azimuth, float elevation,
                                  float radius);
torch::Tensor createTranslationMatrix(float t);
torch::Tensor createPhiRotationMatrix(float phi);
torch::Tensor createThetaRotationMatrix(float theta); */

// needed to convert torch::Tensor to torch::jit::IValue for torchscript
struct Point
{
    float x;
    float y;
    float z;
    Point(float x, float y, float z) : x(x), y(y), z(z) {};
};
torch::Device get_device();
torch::Tensor create_gaussian_kernel(int size, float mean, float std);
torch::Tensor raw2attenuation(torch::Tensor raw, torch::Tensor dists);
torch::Tensor sin_fn(const torch::Tensor &x);
torch::Tensor cos_fn(const torch::Tensor &x);
void accumulate_rays(torch::Dict<std::string, torch::Tensor> &render_results, torch::Dict<std::string, torch::Tensor> batch_render_results);
torch::Tensor generate_random_rays(Point upper_point, Point lower_point, int num_rays);
cv::Mat tensor_to_grayscale_opencv(const torch::Tensor &tensor);
#endif