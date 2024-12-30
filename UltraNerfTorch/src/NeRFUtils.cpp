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
    std::cout << "Accumulating rays" << std::endl;
    // Accumulate results
    for (auto it = batch_render_results.begin(); it != batch_render_results.end(); ++it)
    {
        if (render_results.find(it->key()) == render_results.end())
        {
            std::cout << "Found new key: " << it->key() << std::endl;
            // First time seeing this key, create a list to accumulate
            render_results.insert(it->key(), torch::empty_like(it->value()));
        }
        std::cout << "Accumulating key: " << it->key() << std::endl;
        render_results.find(it->key())->value().add(it->value());
    }
};
torch::Tensor sin_fn(const torch::Tensor &x)
{
    return torch::sin(x);
};
torch::Tensor cos_fn(const torch::Tensor &x)
{
    return torch::cos(x);
};

/**
 * @brief Generate a batch of random rays between two points.
 *
 * The rays are directed towards the x-axis and originate from the upper_point.
 * The y-coordinate of the ray origin is uniformly distributed between the
 * y-coordinate of the upper_point and the y-coordinate of the lower_point.
 *
 * @param upper_point the point with the upper y-coordinate
 * @param lower_point the point with the lower y-coordinate
 * @param num_rays the number of rays to generate
 * @return a tensor of shape (num_rays, 6) containing the rays
 */
torch::Tensor generate_random_rays(Point upper_point, Point lower_point, int num_rays)
{
    torch::Tensor rays = torch::empty({num_rays, 6}, torch::TensorOptions().dtype(torch::kFloat32));
    // test rays are directed towards x axis
    assert(upper_point.z == lower_point.z);
    float increment = (lower_point.y - upper_point.y) / num_rays;
    float y = upper_point.y;
    for (int i = 0; i < num_rays; i++)
    {
        torch::Tensor ray_o = torch::tensor({upper_point.x, y, upper_point.z}, torch::TensorOptions().dtype(torch::kFloat32));
        torch::Tensor ray_d = torch::tensor({lower_point.x, y, upper_point.z}, torch::TensorOptions().dtype(torch::kFloat32));
        torch::Tensor ray = torch::cat({ray_o, ray_d}, /*dim=*/0);
        rays.index_put_({i}, ray);
        y += increment;
    }
    return rays;
}