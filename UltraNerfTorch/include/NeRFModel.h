#ifndef NERFMODEL_H
#define NERFMODEL_H

#include <iostream>
#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#define slots Q_SLOTS
class NeRFModel : public torch::nn::Module
{
public:
    NeRFModel(const torch::Device &device,
              int D = 8, int W = 256,
              int input = 3,
              int inputChViews = 3,
              int outputChannels = 4,
              std::vector<int> skips = {4},
              int embeddingLevel = 6,
              bool useViewDirs = false);
    torch::Tensor forward(const torch::Tensor &inputs);
    torch::Tensor run_network(const torch::Tensor &inputs);
    void load(std::string path);
    torch::Device get_device();
    bool is_initialized();
    torch::Tensor add_positional_encoding(const torch::Tensor &x) const;

private:
    // TODO: maybe use nn module
    torch::nn::Sequential model_;
    const torch::Device &device_;
    bool initialized_ = false;
    int embedding_level_ = 6;
    int D_ = 8;
    int W_ = 256;
    int input_channels_ = 3;
    int input_ch_views_ = 3;
    int output_channels_ = 4;
    std::vector<int> skips_ = {4};
    bool use_view_dirs_ = false;
    torch::nn::ModuleList pts_linears_;
    torch::nn::Linear output_linear_{nullptr};
};
#endif