#include "NeRFModel.h"
#include "NeRFUtils.h"
#include <cmath>

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
      embedder_(Embedder(Embedder::EmbedderConfig{
          .input_dims = 3,
          .include_input = true,
          .max_freq_log2 = 10,
          .num_freqs = 4,
          .log_sampling = true,
          .periodic_fns = {sin_fn, cos_fn},
      })),
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
void NeRFModel::load_weights(const std::string &path)
{
    try
    {
        // Load the PyTorch model weights
        torch::jit::script::Module pytorch_model = torch::jit::load(path);

        // Load weights for pts_linears
        for (size_t i = 0; i < pts_linears_->size(); i++)
        {
            std::string weight_key = "pts_linears." + std::to_string(i) + ".weight";
            std::string bias_key = "pts_linears." + std::to_string(i) + ".bias";

            auto &layer = *std::dynamic_pointer_cast<torch::nn::Linear>(pts_linears_[i]);

            // Load weights and biases
            layer->weight.copy_(pytorch_model.attr(weight_key).toTensor());
            layer->bias.copy_(pytorch_model.attr(bias_key).toTensor());
        }

        // Load weights for output_linear
        output_linear_->weight.copy_(pytorch_model.attr("output_linear.weight").toTensor());
        output_linear_->bias.copy_(pytorch_model.attr("output_linear.bias").toTensor());
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Error loading weights: " << e.msg() << std::endl;
        throw;
    }
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
    return embedder_.embed(x);
}

torch::Device NeRFModel::get_device() { return device_; }
bool NeRFModel::is_initialized() { return initialized_; }

Embedder::Embedder(const EmbedderConfig &kwargs)
    : config(kwargs), out_dim(0)
{
    create_embedding_fn();
}

void Embedder::create_embedding_fn()
{
    embed_fns.clear();
    out_dim = 0;

    // Include input if specified
    if (config.include_input)
    {
        embed_fns.push_back([](const torch::Tensor &x)
                            { return x; });
        out_dim += config.input_dims;
    }

    // Create frequency bands
    torch::Tensor freq_bands;
    if (config.log_sampling)
    {
        // Logarithmically spaced frequencies
        freq_bands = torch::pow(2.0,
                                torch::linspace(0.0, config.max_freq_log2, config.num_freqs));
    }
    else
    {
        // Linearly spaced frequencies between 2^0 and 2^max_freq
        double start = std::pow(2.0, 0.0);
        double end = std::pow(2.0, config.max_freq_log2);
        freq_bands = torch::linspace(start, end, config.num_freqs);
    }

    // Create embedding functions for each frequency and periodic function
    for (int64_t i = 0; i < freq_bands.size(0); ++i)
    {
        double freq = freq_bands[i].item<double>();
        for (const auto &p_fn : config.periodic_fns)
        {
            embed_fns.push_back(
                [p_fn, freq](const torch::Tensor &x)
                {
                    return p_fn(x * freq);
                });
            out_dim += config.input_dims;
        }
    }
}

int Embedder::get_out_dim() const
{
    return out_dim;
}

torch::Tensor Embedder::embed(const torch::Tensor &inputs) const
{
    std::vector<torch::Tensor> outputs;
    outputs.reserve(embed_fns.size());

    for (const auto &fn : embed_fns)
    {
        outputs.push_back(fn(inputs));
    }

    // Concatenate along the last dimension
    return torch::cat(outputs, -1);
}