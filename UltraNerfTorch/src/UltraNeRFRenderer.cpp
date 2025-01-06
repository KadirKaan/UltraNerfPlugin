#include "UltraNeRFRenderer.h"
#include "NeRFUtils.h"

using namespace torch::indexing;

UltraNeRFRenderer::UltraNeRFRenderer(NeRFModel model, int H, int W, float sw, float sh)
    : H{H}, W{W}, sw{sw}, sh{sh}, model_ptr_{std::make_unique<NeRFModel>(model)}
{
    this->gaussian_kernel = create_gaussian_kernel(3, 0.0, 1.0);
};
std::pair<torch::Tensor, torch::Tensor> UltraNeRFRenderer::get_rays(const std::optional<torch::Tensor> &c2w,
                                                                    const std::optional<std::pair<torch::Tensor, torch::Tensor>> &input_rays)
{

    // Validate input: either rays or c2w must be provided
    if (!input_rays.has_value() && !c2w.has_value())
    {
        throw std::invalid_argument("Either rays or c2w must be provided");
    }
    torch::Tensor rays_o, rays_d;
    if (c2w.has_value())
    {
        // Special case to render full image
        auto c = c2w.value();
        auto [o, d] = generate_linear_us_rays(c);
        rays_o = o;
        rays_d = d;
    }
    else
    {
        // Use provided ray batch
        auto [o, d] = input_rays.value();
        rays_o = o;
        rays_d = d;
    }
    return std::pair{rays_o, rays_d};
}

std::vector<torch::Tensor> UltraNeRFRenderer::batchify_rays(torch::Tensor rays_flat)
{
    std::vector<torch::Tensor> ray_batches = std::vector<torch::Tensor>();
    // Iterate through rays in chunks
    for (int i = 0; i < rays_flat.size(0); i += chunk)
    {
        // Get chunk of rays
        int end_idx = std::min(i + chunk, static_cast<int>(rays_flat.size(0)));
        torch::Tensor rays_chunk = rays_flat.slice(0, i, end_idx);
        ray_batches.push_back(rays_chunk);
    }
    return ray_batches;
}
torch::Dict<std::string, torch::Tensor> UltraNeRFRenderer::render_ray_batches(std::vector<torch::Tensor> ray_batches)
{
    torch::Dict<std::string, torch::Tensor> render_results = torch::Dict<std::string, torch::Tensor>();

    for (auto rays_chunk : ray_batches)
    {
        auto raw_ray_chunks = pass_rays_to_nerf(rays_chunk);
        auto rendered_ray_chunks = process_raw_rays(raw_ray_chunks);
        render_results = rendered_ray_chunks;
        // TODO: add batch rendering
        // accumulate_rays(render_results, rendered_ray_chunks);
    }
    return render_results;
    // // Concatenate results for each key
    // torch::Dict<std::string, torch::Tensor> flat_render_results = torch::Dict<std::string, torch::Tensor>();
    // for (auto it = render_results.begin(); it != render_results.end(); ++it)
    // {
    //     flat_render_results.insert(it->key(), torch::cat(it->value(), 0));
    // }
    // return flat_render_results;
}
torch::Dict<std::string, torch::Tensor> UltraNeRFRenderer::render_nerf(
    std::optional<std::pair<torch::Tensor, torch::Tensor>> input_rays = std::nullopt,
    std::optional<torch::Tensor> c2w = std::nullopt)
{

    std::pair<torch::Tensor, torch::Tensor> rays = get_rays(c2w, input_rays);
    torch::Tensor rays_o = rays.first;
    torch::Tensor rays_d = rays.second;
    // Reshape rays
    auto sh = rays_d.sizes();
    rays_o = rays_o.reshape({-1, 3}).to(torch::kFloat32);
    rays_d = rays_d.reshape({-1, 3}).to(torch::kFloat32);

    // Create near and far tensors
    auto near_tensor = torch::ones_like(rays_d.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)})) * near;
    auto far_tensor = torch::ones_like(rays_d.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)})) * far;

    // Concatenate rays information
    torch::Tensor concat_rays = torch::cat({rays_o, rays_d, near_tensor, far_tensor}, /*dim=*/-1);

    auto ray_batches = batchify_rays(concat_rays);
    torch::Dict<std::string, torch::Tensor> render_results = render_ray_batches(ray_batches);
    return render_results;
}

std::pair<torch::Tensor, torch::Tensor> UltraNeRFRenderer::generate_linear_us_rays(const torch::Tensor &c2w)
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
    torch::Tensor origin_rotated = R * origin_base_prim.to(R.device());
    torch::Tensor ray_o_r = torch::sum(origin_rotated, /*dim=*/-1);
    torch::Tensor rays_o = ray_o_r + t;
    // Define direction base and rotate
    torch::Tensor dirs_base = torch::tensor({0., 1., 0.}, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor dirs_r = torch::mv(R, dirs_base.to(R.device()));
    torch::Tensor rays_d = dirs_r.expand_as(rays_o);
    return {rays_o, rays_d};
};

/**
 * Pass a batch of rays to the NeRF model.
 *
 * Given a batch of rays, sample along each ray and evaluate the model at each point.
 * The model is queried with the points in space where the rays intersect the scene.
 *
 * @param ray_batch A tensor of shape `(N_rays, 8)` containing the origin and direction of each ray.
 * @param N_samples The number of samples to take along each ray.
 * @param retraw Whether to use the raw RGB and density values from the model.
 * @param lindisp Whether to use linear disparity (1/depth) instead of real depth.
 *
 * @return A tensor of shape `(N_rays, N_samples, 4)` containing the raw RGB and density values for each sample.
 */
torch::Tensor UltraNeRFRenderer::pass_rays_to_nerf(
    torch::Tensor ray_batch,
    bool retraw,
    bool lindisp)
{
    // Batch size
    int N_rays = ray_batch.size(0);

    // Extract ray origin and direction
    torch::Tensor rays_o = ray_batch.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
    torch::Tensor rays_d = ray_batch.index({torch::indexing::Slice(), torch::indexing::Slice(3, 6)});
    // Extract bounds
    torch::Tensor bounds = ray_batch.index({torch::indexing::Ellipsis, torch::indexing::Slice(6, 8)}).reshape({-1, 1, 2});
    torch::Tensor near = bounds.index({torch::indexing::Ellipsis, 0});
    torch::Tensor far = bounds.index({torch::indexing::Ellipsis, 1});

    // Sampling along rays
    torch::Tensor t_vals = torch::linspace(0., 1., samples_per_ray).to(ray_batch.device());
    torch::Tensor z_vals;

    if (!lindisp)
    {
        z_vals = near * (1. - t_vals) + far * t_vals;
    }
    else
    {
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals);
    }
    z_vals = z_vals.expand({N_rays, samples_per_ray});
    // Points in space to evaluate
    torch::Tensor origin = rays_o.unsqueeze(-2);
    torch::Tensor step = rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1);
    torch::Tensor pts = step + origin;
    // Evaluate model at each point
    torch::Tensor raw = model_ptr_->run_network(pts);
    return raw;
}

// Assuming these are defined elsewhere or need to be added
// Mimics tf.math.cumprod(..., exclusive=True)
torch::Tensor cumprod_exclusive(torch::Tensor tensor)
{
    // Works only for the last dimension (dim=-1)
    const int dim = -1;

    // Compute regular cumprod first (equivalent to tf.math.cumprod(..., exclusive=False))
    torch::Tensor cumprod = torch::cumprod(tensor, dim);

    // "Roll" the elements along dimension 'dim' by 1 element
    cumprod = torch::roll(cumprod, 1, dim);

    // Replace the first element with 1
    // Using indexing to set the first element of the last dimension to 1
    cumprod.index_put_({torch::indexing::Ellipsis, 0}, 1.0);

    return cumprod;
}

torch::Dict<std::string, torch::Tensor> UltraNeRFRenderer::process_raw_rays(torch::Tensor raw)
{
    this->gaussian_kernel = gaussian_kernel;
    // Preprocessing raw tensor
    raw = raw.unsqueeze(0).unsqueeze(1);
    raw = raw.permute({0, 1, 3, 2, 4});
    auto batch_size = raw.size(0);
    auto C = raw.size(1);
    auto W = raw.size(2);
    auto H = raw.size(3);
    auto maps = raw.size(4);

    // Generate z values
    torch::Tensor t_vals = torch::linspace(0.0, 1.0, H).to(raw.device());
    torch::Tensor z_vals = t_vals.expand({batch_size, W, -1}).to(raw.device());

    // Calculate distances
    torch::Tensor dists = torch::abs(
        z_vals.index({torch::indexing::Ellipsis, torch::indexing::Slice(None, -1), torch::indexing::Slice(None, 1)}) -
        z_vals.index({torch::indexing::Ellipsis, torch::indexing::Slice(1, None), torch::indexing::Slice(None, 1)}));
    dists = torch::squeeze(dists);

    auto last_col = dists.index({"...", -1}).unsqueeze(-1).to(raw.device());
    dists = torch::cat({dists, last_col}, -1);

    // Attenuation
    torch::Tensor attenuation_coeff = torch::abs(raw.index({torch::indexing::Ellipsis, 0}));
    torch::Tensor attenuation = raw2attenuation(attenuation_coeff, dists);
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
                                    scatterers_map,
                                    gaussian_kernel.to(raw.device()).unsqueeze(0).unsqueeze(0),
                                    torch::nn::functional::Conv2dFuncOptions()
                                        .stride(1)
                                        .padding(gaussian_kernel.size(0) / 2))
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
    return results;
}
torch::Tensor UltraNeRFRenderer::get_output_data(torch::Dict<std::string, torch::Tensor> output_dict)
{
    return output_dict.at("intensity_map");
}
