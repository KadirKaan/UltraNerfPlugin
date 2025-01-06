#include "NeRFUtils.h"

#include <fstream>
#include <iostream>
#include <string_view>

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
    std::cout << "raw2attenuation" << std::endl;
    std::cout << "raw" << raw.sizes() << std::endl;
    std::cout << "dists" << dists.sizes() << std::endl;
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
            render_results.insert(it->key(), torch::empty_like(it->value()));
        }
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

cv::Mat tensor_to_grayscale_opencv(const torch::Tensor &tensor)
{
    // Ensure tensor is on CPU and contiguous
    auto tensor_cpu = tensor.cpu().contiguous();

    // Convert to uint8 if in float format
    if (tensor_cpu.dtype() == torch::kFloat32)
    {
        // Scale to 0-255 if in range 0-1
        if (tensor_cpu.max().item<float>() <= 1.0)
        {
            tensor_cpu = tensor_cpu * 255.0;
        }
        tensor_cpu = tensor_cpu.clamp(0, 255).to(torch::kUInt8);
    }

    cv::Mat output_mat;

    if (tensor_cpu.dim() == 2)
    {
        // Already grayscale
        output_mat = cv::Mat(
            tensor_cpu.size(0),
            tensor_cpu.size(1),
            CV_8UC1,
            tensor_cpu.data_ptr<unsigned char>());
    }
    else if (tensor_cpu.dim() == 3)
    {
        int channels = tensor_cpu.size(0);

        if (channels == 1)
        {
            // Single channel, just reshape
            tensor_cpu = tensor_cpu.squeeze(0);
            output_mat = cv::Mat(
                tensor_cpu.size(0),
                tensor_cpu.size(1),
                CV_8UC1,
                tensor_cpu.data_ptr<unsigned char>());
        }
        else if (channels == 3 || channels == 4)
        {
            // RGB/RGBA to grayscale
            // First convert to HWC format
            tensor_cpu = tensor_cpu.permute({1, 2, 0});

            // Create temporary RGB Mat
            cv::Mat temp_mat(
                tensor_cpu.size(0),
                tensor_cpu.size(1),
                channels == 3 ? CV_8UC3 : CV_8UC4,
                tensor_cpu.data_ptr<unsigned char>());

            // Convert RGB to BGR if it's a 3-channel image
            if (channels == 3)
            {
                cv::cvtColor(temp_mat, temp_mat, cv::COLOR_RGB2BGR);
            }

            // Convert to grayscale
            cv::cvtColor(temp_mat, output_mat, channels == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
        }
        else
        {
            throw std::runtime_error("Unsupported number of channels");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported tensor dimensions");
    }

    return output_mat.clone();
}