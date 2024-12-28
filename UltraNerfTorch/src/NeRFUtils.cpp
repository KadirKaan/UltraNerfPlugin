#include "NeRFUtils.h"

#include <fstream>
#include <iostream>
torch::Device get_device()
{
    return torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
}

torch::Tensor create_gaussian_kernel(int size, float mean, float std)
{
    float delta_t = 1.0;

    // Create range for x coordinates
    auto x_range = torch::arange(-size, size + 1, torch::TensorOptions().dtype(torch::kFloat32));
    x_range = x_range * delta_t;

    // Create normal distributions
    // Note: Using a custom normal distribution implementation since torch::distributions
    // might not have exact equivalent functionality
    auto normal_pdf = [](float x, float mean, float std) -> float
    {
        float variance = std * std;
        return (1.0 / (std * std::sqrt(2.0 * M_PI))) *
               std::exp(-0.5 * std::pow((x - mean) / std, 2));
    };

    // Calculate probabilities for both distributions
    auto vals_x = torch::empty_like(x_range);
    auto vals_y = torch::empty_like(x_range);

    // Fill probability values
    for (int i = 0; i < x_range.size(0); i++)
    {
        float x = x_range[i].item<float>();
        vals_x[i] = normal_pdf(x, mean, std * 3);
        vals_y[i] = normal_pdf(x, mean, std);
    }

    // Compute outer product (equivalent to einsum("i,j->ij"))
    auto gauss_kernel = torch::outer(vals_x, vals_y);

    // Normalize the kernel
    gauss_kernel = gauss_kernel / torch::sum(gauss_kernel);

    return gauss_kernel;
}
torch::Tensor raw2attenuation(torch::Tensor raw, torch::Tensor dists)
{
    return torch::exp(-raw * dists);
};

void accumulate_rays(torch::Dict<std::string, torch::Tensor> &render_results, torch::Dict<std::string, torch::Tensor> batch_render_results)
{
    // Accumulate results
    for (auto it = batch_render_results.begin(); it != batch_render_results.end(); ++it)
    {
        if (render_results.find(it->key()) == render_results.end())
        {
            // First time seeing this key, create a list to accumulate
            render_results.insert(it->key(), torch::Tensor());
        }
        render_results.find(it->key())->value().add(it->value());
    }
};