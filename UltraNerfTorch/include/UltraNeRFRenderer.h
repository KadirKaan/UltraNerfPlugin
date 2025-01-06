#ifndef ULTRANERFRENDERER_H
#define ULTRANERFRENDERER_H

#include <iostream>
#include <NeRFModel.h>
#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#define slots Q_SLOTS

class UltraNeRFRenderer
{
public:
    explicit UltraNeRFRenderer(NeRFModel model, int H, int W, float sw, float sh);
    virtual torch::Dict<std::string, torch::Tensor> render_nerf(const std::optional<std::pair<torch::Tensor, torch::Tensor>> rays,
                                                                const std::optional<torch::Tensor> c2w);
    virtual torch::Tensor get_output_data(torch::Dict<std::string, torch::Tensor> output_dict);

private:
    std::pair<torch::Tensor, torch::Tensor> generate_linear_us_rays(const torch::Tensor &c2w);
    torch::Dict<std::string, torch::Tensor> render_ray_batches(std::vector<torch::Tensor> ray_batches);
    virtual std::pair<torch::Tensor, torch::Tensor> get_rays(const std::optional<torch::Tensor> &c2w,
                                                             const std::optional<std::pair<torch::Tensor, torch::Tensor>> &rays);
    virtual std::vector<torch::Tensor> batchify_rays(torch::Tensor rays_flat);
    virtual torch::Tensor pass_rays_to_nerf(torch::Tensor batched_rays,
                                            bool retraw = false,
                                            bool lindisp = false);
    virtual torch::Dict<std::string, torch::Tensor> process_raw_rays(torch::Tensor raw);
    int chunk = 1024 * 32;
    torch::Tensor gaussian_kernel;
    std::unique_ptr<NeRFModel> model_ptr_;
    std::unique_ptr<torch::Dict<std::string, torch::Tensor>> result_dict_ptr_;
    int H;
    int W;
    float sw;
    float sh;
    float near = 0.0;
    float far = 55.0 * 0.001;
    int samples_per_ray = 512;
};
#endif