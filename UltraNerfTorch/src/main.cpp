#include <torch/torch.h>
#include <iostream>
#include <NeRFModel.h>
#include <UltraNeRFRenderer.h>
#include <NeRFUtils.h>
int main()
{
    NeRFModel model = NeRFModel(get_device());
    UltraNeRFRenderer renderer = UltraNeRFRenderer(model, 512, 512, 1.0, 1.0);
    torch::Tensor rays = generate_random_rays({0.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, 512);
    std::pair<torch::Tensor, torch::Tensor> ray_o_d_pairs = std::make_pair(rays.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}),
                                                                           rays.index({torch::indexing::Slice(), torch::indexing::Slice(3, 6)}));
    renderer.render_nerf(std::optional<std::pair<torch::Tensor, torch::Tensor>>(ray_o_d_pairs), std::nullopt);
    torch::Tensor output = renderer.get_output_data(torch::Dict<std::string, torch::Tensor>());
    std::cout << output << std::endl;
    return 0;
}