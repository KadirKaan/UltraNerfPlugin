#ifndef NERFRENDERER_H
#define NERFRENDERER_H

#include <iostream>
#include <NeRFModel.h>
#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#define slots Q_SLOTS
class NeRFRenderer
{
public:
    NeRFRenderer();
    NeRFRenderer(NeRFModel &model, int H, int W, float focal);
    torch::Tensor render(const torch::Tensor &pose, bool randomize = false,
                         float startDistance = 2.0f, float endDistance = 5.0f,
                         int nSamples = 64, int batchSize = 64000) const;

private:
    typedef std::tuple<torch::Tensor, torch::Tensor> RayData;

    NeRFModel &model;
    int H;
    int W;
    float focal;

    RayData getRays(const torch::Tensor &pose) const;
    torch::Tensor renderRays(const RayData &rays, bool randomize,
                             float startDistance, float endDistance,
                             int nSamples, int batchSize) const;
};
#endif