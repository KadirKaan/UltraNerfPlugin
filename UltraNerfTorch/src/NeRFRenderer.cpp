#include "NeRFRenderer.h"
#include "NeRFUtils.h"

using namespace torch::indexing;

NeRFRenderer::NeRFRenderer(NeRFModel &model)
{
    network_fn_ = model.get_network_fn();
    network_query_fn_ = model.get_network_query_fn();
    // todo get network and network_query from model
    // maybe remove model from NeRFRenderer and pass it as a parameter
}

torch::Dict<std::string, torch::Tensor> NeRFRenderer::batchify_rays(
    torch::Tensor rays_flat,
    int chunk = 1024 * 32,
    int N_samples = 1)
{
    torch::Dict<std::string, torch::Tensor> all_ret;

    // Iterate through rays in chunks
    for (int i = 0; i < rays_flat.size(0); i += chunk)
    {
        // Get chunk of rays
        int end_idx = std::min(i + chunk, static_cast<int>(rays_flat.size(0)));
        torch::Tensor rays_chunk = rays_flat.slice(0, i, end_idx);
        torch::Dict<std::string, std::vector<torch::Tensor>> all_ret;

        // Call render_rays_us with the chunk
        auto ret = render_rays_us(rays_chunk, N_samples);

        // Accumulate results
        for (auto it = ret.begin(); it != ret.end(); ++it)
        {
            if (all_ret.find(it->key()) == all_ret.end())
            {
                // First time seeing this key, create a list to accumulate
                all_ret.insert(it->key(), std::vector<torch::Tensor>());
            }
            all_ret.find(it->key())->value().push_back(it->value());
        }
    }

    // Concatenate results for each key
    torch::Dict<std::string, torch::Tensor> final_ret;
    for (auto it = all_ret.begin(); it != all_ret.end(); ++it)
    {
        final_ret.insert(it->key(), torch::cat(it->value(), 0));
    }
    return final_ret;
}
// TODO: 1 to 1 mapping of funcs in python, format them later
std::pair<torch::Tensor, torch::Tensor> get_rays_us_linear(
    int H, int W, float sw, float sh, torch::Tensor c2w)
{
    // Extract translation and rotation from camera-to-world matrix
    torch::Tensor t = c2w.index({torch::indexing::Slice(0, 3), -1});
    torch::Tensor R = c2w.index({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)});

    // Create a range of x-values
    torch::Tensor x = torch::arange(-W / 2, W / 2, torch::TensorOptions().dtype(torch::kFloat32)) * sw;
    torch::Tensor y = torch::zeros_like(x);
    torch::Tensor z = torch::zeros_like(x);

    // Stack and prepare the origin base
    torch::Tensor origin_base = torch::stack({x, y, z}, /*dim=*/1);
    torch::Tensor origin_base_prim = origin_base.unsqueeze(/*dim=*/-2);

    // Rotate the origin base
    torch::Tensor origin_rotated = R * origin_base_prim;
    torch::Tensor ray_o_r = torch::sum(origin_rotated, /*dim=*/-1);
    torch::Tensor rays_o = ray_o_r + t;

    // Define direction base and rotate
    torch::Tensor dirs_base = torch::tensor({0., 1., 0.}, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor dirs_r = torch::mv(R, dirs_base);
    torch::Tensor rays_d = dirs_r.expand_as(rays_o);

    return {rays_o, rays_d};
};

torch::Dict<std::string, torch::Tensor> NeRFRenderer::render_rays_us(
    torch::Tensor ray_batch,
    int N_samples,
    bool retraw = false,
    bool lindisp = false)
{
    // Helper function to transform model predictions
    auto raw2outputs = [](torch::Tensor raw)
    {
        return render_method_ultra_nerf(raw);
    };

    // Batch size
    int N_rays = ray_batch.size(0);

    // Extract ray origin and direction
    torch::Tensor rays_o = ray_batch.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
    torch::Tensor rays_d = ray_batch.index({torch::indexing::Slice(), torch::indexing::Slice(3, 6)});

    // Extract viewing direction
    std::optional<torch::Tensor> viewdirs = std::nullopt;
    if (ray_batch.size(-1) > 8)
    {
        viewdirs = ray_batch.index({torch::indexing::Slice(), torch::indexing::Slice(-3, std::nullopt)});
    }

    // Extract bounds
    torch::Tensor bounds = ray_batch.index({torch::indexing::Ellipsis, torch::indexing::Slice(6, 8)}).reshape({-1, 1, 2});
    torch::Tensor near = bounds.index({torch::indexing::Ellipsis, 0});
    torch::Tensor far = bounds.index({torch::indexing::Ellipsis, 1});

    // Sampling along rays
    torch::Tensor t_vals = torch::linspace(0., 1., N_samples);
    torch::Tensor z_vals;

    if (!lindisp)
    {
        z_vals = near * (1. - t_vals) + far * t_vals;
    }
    else
    {
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals);
    }
    z_vals = z_vals.expand({N_rays, N_samples});

    // Points in space to evaluate
    torch::Tensor origin = rays_o.unsqueeze(-2);
    torch::Tensor step = rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1);
    torch::Tensor pts = step + origin;

    // Evaluate model at each point
    torch::Tensor raw = this->network_query_fn_(pts, network_fn_);

    // Transform raw predictions
    auto ret = raw2outputs(raw);

    return ret;
}

torch::Dict<std::string, torch::Tensor> NeRFRenderer::render_us(
    int H, int W,
    float sw, float sh,
    int chunk = 1024 * 32,
    std::optional<std::pair<torch::Tensor, torch::Tensor>> rays = std::nullopt,
    std::optional<std::vector<torch::Tensor>> c2w = std::nullopt,
    float near = 0.0,
    float far = 55.0 * 0.001)
{
    // Validate input: either rays or c2w must be provided
    if (!rays.has_value() && !c2w.has_value())
    {
        throw std::invalid_argument("Either rays or c2w must be provided");
    }

    torch::Tensor rays_o, rays_d;

    if (c2w.has_value())
    {
        // Special case to render full image
        for (const auto &c : c2w.value())
        {
            auto [o, d] = get_rays_us_linear(H, W, sw, sh, c);

            if (!rays_o.defined())
            {
                rays_o = o;
                rays_d = d;
            }
            else
            {
                rays_o = torch::cat({rays_o, o}, 0);
                rays_d = torch::cat({rays_d, d}, 0);
            }
        }
    }
    else
    {
        // Use provided ray batch
        auto [o, d] = rays.value();
        rays_o = o;
        rays_d = d;
    }

    // Reshape rays
    auto sh = rays_d.sizes();
    rays_o = rays_o.reshape({-1, 3}).to(torch::kFloat32);
    rays_d = rays_d.reshape({-1, 3}).to(torch::kFloat32);

    // Create near and far tensors
    auto near_tensor = torch::ones_like(rays_d.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)})) * near;
    auto far_tensor = torch::ones_like(rays_d.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)})) * far;

    // Concatenate rays information
    torch::Tensor concat_rays = torch::cat({rays_o, rays_d, near_tensor, far_tensor}, /*dim=*/-1);

    auto all_ret = batchify_rays(concat_rays, chunk);
    return all_ret;
}

// Assuming these are defined elsewhere or need to be added
torch::Tensor cumprod_exclusive(torch::Tensor tensor);
torch::Tensor g_kernel; // Global convolution kernel

torch::Dict<std::string, torch::Tensor> render_method_ultra_nerf(torch::Tensor raw)
{
    // Helper function
    auto raw2attention = [](torch::Tensor raw, torch::Tensor dists)
    {
        return torch::exp(-raw * dists);
    };

    // Preprocessing raw tensor
    raw = raw.unsqueeze(0).unsqueeze(1);
    raw = raw.permute({0, 1, 3, 2, 4});

    auto batch_size = raw.size(0);
    auto C = raw.size(1);
    auto W = raw.size(2);
    auto H = raw.size(3);
    auto maps = raw.size(4);

    // Generate z values
    torch::Tensor t_vals = torch::linspace(0.0, 1.0, H).to(torch::kCUDA);
    torch::Tensor z_vals = t_vals.expand({batch_size, W, -1});

    // Calculate distances
    torch::Tensor dists = torch::abs(
        z_vals.index({torch::indexing::Ellipsis, torch::indexing::Slice(None, -1), torch::indexing::Slice(None, 1)}) -
        z_vals.index({torch::indexing::Ellipsis, torch::indexing::Slice(1, None), torch::indexing::Slice(None, 1)}));
    dists = torch::squeeze(dists);
    dists = torch::cat({dists, dists.index({torch::indexing::Slice(), torch::indexing::Slice(-1, None)})}, -1);

    // Attenuation
    torch::Tensor attenuation_coeff = torch::abs(raw.index({torch::indexing::Ellipsis, 0}));
    torch::Tensor attenuation = raw2attention(attenuation_coeff, dists);
    attenuation = attenuation.permute({0, 1, 3, 2});
    torch::Tensor attenuation_total = cumprod_exclusive(attenuation);
    attenuation_total = attenuation_total.permute({0, 1, 3, 2});

    // Reflection
    torch::Tensor prob_border = torch::sigmoid(raw.index({torch::indexing::Ellipsis, 2}));

    // Note: LibTorch doesn't have direct RelaxedBernoulli equivalent
    // This is a simplified approximation
    torch::Tensor b_prob = torch::bernoulli(prob_border);

    torch::Tensor reflection_coeff = torch::sigmoid(raw.index({torch::indexing::Ellipsis, 1}));
    torch::Tensor reflection_transmission = 1. - reflection_coeff * b_prob;
    reflection_transmission = reflection_transmission.permute({0, 1, 3, 2});
    torch::Tensor reflection_total = cumprod_exclusive(reflection_transmission);
    reflection_total = reflection_total.permute({0, 1, 3, 2});

    // Backscattering
    torch::Tensor density_coeff = torch::sigmoid(raw.index({torch::indexing::Ellipsis, 3}));

    // Probabilistic sampling (approximation)
    torch::Tensor scatterers_density = torch::bernoulli(density_coeff);

    torch::Tensor amplitude = torch::sigmoid(raw.index({torch::indexing::Ellipsis, 4}));
    torch::Tensor scatterers_map = scatterers_density * amplitude;

    // Convolution (assuming g_kernel is available)
    torch::Tensor psf_scatter = torch::nn::functional::conv2d(
                                    scatterers_map.unsqueeze(0).unsqueeze(0),
                                    g_kernel.unsqueeze(0).unsqueeze(0),
                                    torch::nn::functional::Conv2dFuncOptions().stride(1).padding(1))
                                    .squeeze();

    // Compute confidence maps and final intensity
    torch::Tensor confidence_maps = attenuation_total * reflection_total;
    torch::Tensor b = confidence_maps * psf_scatter;
    torch::Tensor r = confidence_maps * reflection_coeff;

    // Simplified intensity calculation (removed amplification)
    torch::Tensor intensity_map = b + r;

    torch::Dict<std::string, torch::Tensor> results = torch::Dict<std::string, torch::Tensor>();

    // TODO: why isnt default constructor working
    results.insert("intensity_map", intensity_map);
    results.insert("attenuation_coeff", attenuation_coeff);
    results.insert("reflection_coeff", reflection_coeff);
    results.insert("attenuation_total", attenuation_total);
    results.insert("reflection_total", reflection_total);
    results.insert("scatterers_density", scatterers_density);
    results.insert("scatterers_density_coeff", density_coeff);
    results.insert("scatter_amplitude", amplitude);
    results.insert("b", b);
    results.insert("r", r);
    results.insert("confidence_maps", confidence_maps);
}