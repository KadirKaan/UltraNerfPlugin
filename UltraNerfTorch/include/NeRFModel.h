#ifndef NERFMODEL_H
#define NERFMODEL_H

#include <iostream>
#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <functional>
#include <cmath>
#define slots Q_SLOTS

class Embedder
{
public:
    struct EmbedderConfig
    {
        int input_dims;
        bool include_input;
        int max_freq_log2;
        int num_freqs;
        bool log_sampling;
        std::vector<std::function<torch::Tensor(const torch::Tensor &)>> periodic_fns;
    };

    explicit Embedder(const EmbedderConfig &kwargs);

    void create_embedding_fn();
    int get_out_dim() const;
    torch::Tensor embed(const torch::Tensor &inputs) const;

private:
    EmbedderConfig config;
    std::vector<std::function<torch::Tensor(const torch::Tensor &)>> embed_fns;
    int out_dim;
};
class NeRFModel : public torch::nn::Module
{
public:
    NeRFModel(const torch::Device &device,
              int D = 8, int W = 256,
              int inputChViews = 3,
              int outputChannels = 5,
              std::vector<int> skips = {4},
              int embeddingLevel = 10,
              bool useViewDirs = false);
    torch::Tensor forward(const torch::Tensor &inputs);
    torch::Tensor run_network(const torch::Tensor &inputs);
    void load_weights(const std::string &path);
    torch::Device get_device();
    bool is_initialized();
    torch::Tensor add_positional_encoding(const torch::Tensor &x) const;

private:
    // TODO: this is somehow broken
    const torch::Device &device_;
    bool initialized_ = false;
    int embedding_level_;
    int D_;
    int W_;
    int input_channels_;
    int input_ch_views_;
    int output_channels_;
    std::vector<int> skips_ = {4};
    bool use_view_dirs_ = false;
    torch::nn::ModuleList pts_linears_;
    torch::nn::Linear output_linear_{nullptr};
    const Embedder embedder_;
};
#endif