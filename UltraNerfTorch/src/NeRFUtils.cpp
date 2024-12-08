#include "NeRFUtils.h"

#include <fstream>
#include <iostream>
/* #include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
void saveImage(const torch::Tensor &tensor,
               const std::filesystem::path &filePath)
{
    // Assuming the input tensor is a 3-channel (HxWx3) image in the range [0, 1]
    auto height = tensor.size(0);
    auto width = tensor.size(1);
    auto max = tensor.max();
    auto min = tensor.min();
    // auto tensorNormalized = tensor.mul(255)
    auto tensorNormalized = ((tensor - min) / (max - min))
                                .mul(255)
                                .clamp(0, 255)
                                .to(torch::kU8)
                                .to(torch::kCPU)
                                .flatten()
                                .contiguous();
    cv::Mat image(cv::Size(width, height), CV_8UC3, tensorNormalized.data_ptr());
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    cv::imwrite(filePath.string(), image);
}

void renderAndSaveOrbitViews(const NeRFRenderer &renderer, int nFrames,
                             const std::filesystem::path &outputFolder,
                             float radius, float startDistance,
                             float endDistance, int nSamples)
{
    float elevation = -30.0f;

    for (int i = 0; i < nFrames; i++)
    {
        float azimuth = static_cast<float>(i) * 360.0f / nFrames;
        auto pose = createSphericalPose(azimuth, elevation, radius);

        auto renderedImage =
            renderer.render(pose, false, startDistance, endDistance, nSamples);

        std::string file_path =
            outputFolder / ("frame_" + std::to_string(i) + ".png");
        saveImage(renderedImage, file_path);
    }
}

torch::Tensor createSphericalPose(float azimuth, float elevation,
                                  float radius)
{
    float phi = elevation * (M_PI / 180.0f);
    float theta = azimuth * (M_PI / 180.0f);

    torch::Tensor c2w = createTranslationMatrix(radius);
    c2w = createPhiRotationMatrix(phi).matmul(c2w);
    c2w = createThetaRotationMatrix(theta).matmul(c2w);
    c2w = torch::tensor({{-1.0f, 0.0f, 0.0f, 0.0f},
                         {0.0f, 0.0f, 1.0f, 0.0f},
                         {0.0f, 1.0f, 0.0f, 0.0f},
                         {0.0f, 0.0f, 0.0f, 1.0f}})
              .matmul(c2w);

    return c2w;
}

torch::Tensor createTranslationMatrix(float t)
{
    torch::Tensor tMat = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f},
                                        {0.0f, 1.0f, 0.0f, 0.0f},
                                        {0.0f, 0.0f, 1.0f, t},
                                        {0.0f, 0.0f, 0.0f, 1.0f}});
    return tMat;
}

torch::Tensor createPhiRotationMatrix(float phi)
{
    torch::Tensor phiMat =
        torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f},
                       {0.0f, std::cos(phi), -std::sin(phi), 0.0f},
                       {0.0f, std::sin(phi), std::cos(phi), 0.0f},
                       {0.0f, 0.0f, 0.0f, 1.0f}});
    return phiMat;
}

torch::Tensor createThetaRotationMatrix(float theta)
{
    torch::Tensor thetaMat =
        torch::tensor({{std::cos(theta), 0.0f, -std::sin(theta), 0.0f},
                       {0.0f, 1.0f, 0.0f, 0.0f},
                       {std::sin(theta), 0.0f, std::cos(theta), 0.0f},
                       {0.0f, 0.0f, 0.0f, 1.0f}});
    return thetaMat;
} */
torch::Device getDevice()
{
    return torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
}