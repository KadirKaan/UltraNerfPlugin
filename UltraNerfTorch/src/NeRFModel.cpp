#include "NeRFModel.h"
#include "NeRFUtils.h"

NeRFModel::NeRFModel(const torch::Device &device, int D, int W, int embeddingLevel)
    : device_(device), embeddingLevel_(embeddingLevel)
{
    // Create FFN
    auto inputDim = 3 + 3 * 2 * embeddingLevel_;
    model_->push_back(torch::nn::Linear(inputDim, W));
    model_->push_back(torch::nn::Functional(torch::relu));
    for (int i = 0; i < D - 2; i++)
    {
        model_->push_back(torch::nn::Linear(W, W));
        model_->push_back(torch::nn::Functional(torch::relu));
    }
    model_->push_back(torch::nn::Linear(W, 4));
    model_->to(device);
    register_module("model", model_);
    this->initialized_ = false;
    this->to(device_);
}

torch::Tensor NeRFModel::forward(torch::Tensor &inputs)
{
    torch::Tensor output = model_->forward(inputs);
    return output;
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
torch::Tensor NeRFModel::addPositionalEncoding(const torch::Tensor &x) const
{
    std::vector<torch::Tensor> enc = {x};
    for (int i = 0; i < embeddingLevel_; i++)
    {
        enc.push_back(torch::sin(std::pow(2.0f, i) * x));
        enc.push_back(torch::cos(std::pow(2.0f, i) * x));
    }
    return torch::cat(enc, -1);
}

torch::Device NeRFModel::getDevice() { return device_; }
bool NeRFModel::isInitialized() { return initialized_; }