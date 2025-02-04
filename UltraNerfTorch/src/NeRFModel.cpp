#include "NeRFModel.h"
#include "NeRFUtils.h"
#include <cmath>

NeRFModel::NeRFModel(const torch::Device &device,
                     int D, int W,
                     int input_ch_views,
                     int output_channels,
                     std::vector<int> skips,
                     int embedding_level,
                     bool use_view_dirs)
    : device_(device),
      embedding_level_(embedding_level),
      D_(D), W_(W),
      input_ch_views_(input_ch_views),
      output_channels_(output_channels),
      skips_(skips),
      pts_linears_(register_module("pts_linears", torch::nn::ModuleList())),
      // TODO: get embedder config values from somewhere
      embedder_(Embedder(Embedder::EmbedderConfig{
          .input_dims = 3,
          .include_input = true,
          .max_freq_log2 = embedding_level_ - 1,
          .num_freqs = embedding_level_,
          .log_sampling = true,
          .periodic_fns = {NeRFUtils::sin_fn, NeRFUtils::cos_fn},
      })),
      torch::nn::Module("NeRFModel")
{
    this->to(device_);
    // Initialize pts_linears ModuleList
    // First layer: input_ch -> W
    input_channels_ = embedder_.get_out_dim();
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
            torch::nn::init::uniform_(linear->weight, -0.05, 0.05).to(get_device());
            torch::nn::init::uniform_(linear->bias, -0.05, 0.05).to(get_device());
        }
    }

    torch::nn::init::uniform_(output_linear_->weight, -0.05, 0.05).to(get_device());
    torch::nn::init::uniform_(output_linear_->bias, -0.05, 0.05).to(get_device());
};
torch::Tensor NeRFModel::forward(const torch::Tensor &x)
{

    auto h = x.clone();
    int index = 0;
    for (const auto &layer : pts_linears_->children())
    {
        // Use layer here
        if (auto *linear = layer.get()->as<torch::nn::Linear>())
        {
            h = linear->forward(h.to(linear->weight.device()));
        }
        h = torch::relu(h);
        if (std::find(skips_.begin(), skips_.end(), index) != skips_.end())
        {
            h = torch::cat({x.to(h.device()), h}, -1);
        }
        index++;
    }
    return output_linear_(h.to(output_linear_->weight.device()));
}

torch::Tensor NeRFModel::run_network(const torch::Tensor &x)
{
    auto input_sizes = x.sizes().vec();
    int64_t last_dim = input_sizes.back();
    input_sizes.pop_back();
    // Flatten inputs
    auto inputs_flat = x.reshape({-1, last_dim});
    torch::Tensor encoded_inputs = add_positional_encoding(inputs_flat);
    input_sizes.push_back(output_channels_);
    torch::Tensor outputs = forward(encoded_inputs).reshape(input_sizes);
    return outputs;
}
void NeRFModel::load_weights(const std::string &path)
{
    // Load the PyTorch model weights
    torch::jit::script::Module pytorch_model = torch::jit::load(path);

    size_t i = 0;

    std::map<std::string, torch::Tensor> params = std::map<std::string, torch::Tensor>();
    for (const auto param : pytorch_model.named_parameters())
    {
        params.insert({param.name, param.value});
    }
    for (const auto &module : pts_linears_->children())
    {
        if (i >= D_)
        {
            break;
        }

        if (auto *linear = module.get()->as<torch::nn::Linear>())
        {
            // Load weights and biases
            std::string weight_key = "pts_linears." + std::to_string(i) + ".weight";
            std::string bias_key = "pts_linears." + std::to_string(i) + ".bias";

            linear->weight = params[weight_key].clone().to(get_device());
            linear->bias = params[bias_key].clone().to(get_device());
        }
        i++;
    }

    // // // Load weights for output_linear
    output_linear_->weight = params["output_linear.weight"].clone().to(get_device());
    output_linear_->bias = params["output_linear.bias"].clone().to(get_device());
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