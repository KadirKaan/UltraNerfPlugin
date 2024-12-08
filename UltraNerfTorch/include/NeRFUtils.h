#ifndef NERF_UTILS_H
#define NERF_UTILS_H

#include <filesystem>
#include "NeRFRenderer.h"
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
torch::Device getDevice();
#endif