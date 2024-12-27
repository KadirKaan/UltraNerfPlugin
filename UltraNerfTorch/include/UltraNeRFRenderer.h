#ifndef ULTRANERFRENDERER_H
#define ULTRANERFRENDERER_H

#include <iostream>
#include <NeRFModel.h>
#include <NeRFRenderer.h>
#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#define slots Q_SLOTS

class UltraNeRFRenderer : public NeRFRenderer
{
public:
    UltraNeRFRenderer(NeRFModel &model, int H, int W, float sw, float sh) : NeRFRenderer(model, H, W, sw, sh);
    virtual std::pair<torch::Tensor, torch::Tensor> get_rays(const std::optional<std::vector<torch::Tensor>> &c2w,
                                                             const std::optional<std::pair<torch::Tensor, torch::Tensor>> &rays);
    virtual std::vector<torch::Tensor> batchify_rays(torch::Tensor rays_flat, int N_samples = 1);
    virtual torch::Tensor pass_rays_to_nerf(torch::Tensor batched_rays, int N_samples,
                                            bool retraw = false,
                                            bool lindisp = false);
    virtual torch::Dict<std::string, torch::Tensor> process_raw_rays(torch::Tensor raw);
    virtual torch::Tensor get_output_data(torch::Dict<std::string, torch::Tensor> output_dict);
    virtual torch::Dict<std::string, torch::Tensor> render_nerf(const std::optional<std::pair<torch::Tensor, torch::Tensor>> rays,
                                                                const std::optional<std::vector<torch::Tensor>> c2w);

private:
    std::pair<torch::Tensor, torch::Tensor> UltraNeRFRenderer::generate_linear_us_rays(const torch::Tensor &c2w);
    torch::Dict<std::string, torch::Tensor> UltraNeRFRenderer::render_ray_batches(std::vector<torch::Tensor> ray_batches, int N_samples = 1);
    int chunk = 1024 * 32;
    float near = 0.0;
    float far = 55.0 * 0.001;
    torch::Tensor gaussian_kernel;
};
#endif