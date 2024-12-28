#ifndef NERFRENDERER_H
#define NERFRENDERER_H

#include <iostream>
#include <NeRFModel.h>
#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#define slots Q_SLOTS

// Rendering interface
class NeRFRenderer
{
public:
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
    virtual NeRFModel get_model() { return model_; };
    NeRFRenderer(NeRFModel &model, int H, int W, float sw, float sh) : model_(model), H(H), W(W), sw(sw), sh(sh) {};
    NeRFRenderer();

protected:
    NeRFModel model_;
    int H;
    int W;
    int chunk = 1024 * 32;
    float sw;
    float sh;
    float near = 0.0;
    float far = 55.0 * 0.001;
};
#endif