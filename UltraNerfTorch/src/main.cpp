#include <torch/torch.h>
#include <iostream>
#include <NeRFModel.h>
#include <UltraNeRFRenderer.h>
#include <NeRFUtils.h>
int main()
{
    NeRFModel model = NeRFModel(get_device());
    UltraNeRFRenderer renderer = UltraNeRFRenderer(model, 512, 512, 1.0, 1.0);
    renderer.render_nerf(std::pair<torch::Tensor, torch::Tensor>(torch::rand({1, 3}), torch::rand({1, 3})), std::nullopt);
    torch::Tensor output = renderer.get_output_data(torch::Dict<std::string, torch::Tensor>());
    std::cout << output << std::endl;
    return 0;
}