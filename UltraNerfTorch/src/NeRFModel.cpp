#include "NeRFModel.h"
#include "NeRFUtils.h"

NeRFModel::NeRFModel(const torch::Device &device,
                     int D, int W,
                     int input_channels,
                     int input_ch_views,
                     int output_channels,
                     std::vector<int> skips,
                     int embedding_level,
                     bool use_view_dirs)
    : device_(device),
      embedding_level_(embedding_level),
      D_(D), W_(W),
      input_channels_(input_channels),
      input_ch_views_(input_ch_views),
      output_channels_(output_channels),
      skips_(skips),
      pts_linears_(register_module("pts_linears", torch::nn::ModuleList())),
      torch::nn::Module("NeRFModel")
{
    // Initialize pts_linears ModuleList
    // First layer: input_ch -> W
    pts_linears_->push_back(register_module("pts_linear_0",
                                            torch::nn::Linear(torch::nn::LinearOptions(input_channels_, W_))));

    // Remaining layers
    for (int i = 0; i < D_ - 1; i++)
    {
        auto found = std::find(skips_.begin(), skips_.end(), i) != skips_.end();
        int input_size = found ? W_ + input_channels_ : W_;

        pts_linears_->push_back(register_module("pts_linear_" + std::to_string(i + 1),
                                                torch::nn::Linear(torch::nn::LinearOptions(input_size, W_))));
    }

    // Output layer
    output_linear_ = register_module("output_linear",
                                     torch::nn::Linear(torch::nn::LinearOptions(W_, output_channels_)));

    // Initialize weights and biases uniformly
    for (const auto &module : pts_linears_->children())
    {
        if (auto *linear = module.get()->as<torch::nn::Linear>())
        {
            torch::nn::init::uniform_(linear->weight, -0.05, 0.05);
            torch::nn::init::uniform_(linear->bias, -0.05, 0.05);
        }
    }

    torch::nn::init::uniform_(output_linear_->weight, -0.05, 0.05);
    torch::nn::init::uniform_(output_linear_->bias, -0.05, 0.05);
};
torch::Tensor NeRFModel::forward(const torch::Tensor &x)
{
    auto h = x;
    for (size_t i = 0; i < pts_linears_->size(); i++)
    {
        auto &layer = *std::dynamic_pointer_cast<torch::nn::Linear>(pts_linears_[i]);
        h = layer(h);
        h = torch::relu(h);

        if (std::find(skips_.begin(), skips_.end(), i) != skips_.end())
        {
            h = torch::cat({x, h}, -1);
        }
    }
    return output_linear_(h);
}

torch::Tensor NeRFModel::run_network(const torch::Tensor &x)
{
    return forward(add_positional_encoding(x));
}
void NeRFModel::load(std::string path)
{
    torch::load(this->model_, path);
    this->initialized_ = true;
}

/**
 * Adds positional encoding to a given tensor.
 *
 * This function adds sine and cosine terms of increasing frequency to the input tensor.
 * The number of terms is controlled by the embeddingLevel member variable.
 *
 * @param x The input tensor.
 * @return The tensor with added positional encoding.
 */
torch::Tensor NeRFModel::add_positional_encoding(const torch::Tensor &x) const
{
    std::vector<torch::Tensor> enc = {x};
    for (int i = 0; i < embedding_level_; i++)
    {
        enc.push_back(torch::sin(std::pow(2.0f, i) * x));
        enc.push_back(torch::cos(std::pow(2.0f, i) * x));
    }
    return torch::cat(enc, -1);
}

torch::Device NeRFModel::get_device() { return device_; }
bool NeRFModel::is_initialized() { return initialized_; }