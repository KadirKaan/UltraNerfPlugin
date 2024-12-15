#ifndef NERFRENDERER_H
#define NERFRENDERER_H

#include <iostream>
#include <NeRFModel.h>
#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#define slots Q_SLOTS

using NetworkFn = std::function<torch::Tensor(torch::Tensor)>;
using NetworkQueryFn = std::function<torch::Tensor(torch::Tensor, NetworkFn)>;
class NeRFRenderer
{
public:
    NeRFRenderer();
    NeRFRenderer(NeRFModel &model);
    torch::Dict<std::string, torch::Tensor> render_rays_us(torch::Tensor ray_batch, int N_samples, bool retraw, bool lindisp);
    torch::Dict<std::string, torch::Tensor> batchify_rays(torch::Tensor rays_flat, int chunk, int N_samples);
    torch::Dict<std::string, torch::Tensor> render_us(int H, int W, float sw, float sh, int chunk, std::optional<std::pair<torch::Tensor, torch::Tensor>> rays, std::optional<std::vector<torch::Tensor>> c2w, float near, float far);

private:
    NetworkFn network_fn_;
    NetworkQueryFn network_query_fn_;
};
#endif