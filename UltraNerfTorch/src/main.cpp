#include <torch/torch.h>
#include <iostream>
#include <NeRFModel.h>
#include <UltraNeRFRenderer.h>
#include <NeRFUtils.h>
#include <opencv2/opencv.hpp>
int main()
{
    torch::manual_seed(0);
    NeRFModel model = NeRFModel(get_device());
    model.load_weights("/home/kkaan/Project/UltraNerfPlugin/models/network_fn086000.pt");
    UltraNeRFRenderer renderer = UltraNeRFRenderer(model, 512, 512, 1.0, 1.0);
    torch::Tensor rays = generate_random_rays({0.0, 0.0, 0.0}, {1024.0, 0.0, 0.0}, 512);
    torch::Tensor c2w = torch::tensor({-0.9998, 0.0131, 0.0160, 0.0726,
                                       0.0138, 0.9988, 0.0462, -0.0445,
                                       -0.0154, 0.0464, -0.9988, 0.0538,
                                       0.0000, 0.0000, 0.0000, 1.0000},
                                      torch::kFloat32)
                            .to(get_device())
                            .reshape({1, 4, 4});
    std::cout << c2w << std::endl;
    std::pair<torch::Tensor, torch::Tensor> ray_o_d_pairs = std::make_pair(rays.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}),
                                                                           rays.index({torch::indexing::Slice(), torch::indexing::Slice(3, 6)}));
    torch::Dict<std::string, torch::Tensor> render_results = renderer.render_nerf(std::optional<std::pair<torch::Tensor, torch::Tensor>>(ray_o_d_pairs), std::nullopt);
    torch::Tensor output = renderer.get_output_data(render_results);
    // TODO: fix intensity map incorrectness
    cv::Mat gray_image2 = tensor_to_grayscale_opencv(output.squeeze());
    cv::imshow("Grayscale Image", gray_image2);
    cv::waitKey(0);
    return 0;
}