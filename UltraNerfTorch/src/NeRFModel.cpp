#include "NeRFModel.h"
#include "NeRFUtils.h"

void NeRFModel::registerModule(torch::jit::script::Module module)
{
    torch::cuda::is_available() ? (this->device = torch::Device(torch::kCUDA)) : (this->device = torch::Device(torch::kCPU));
    module.to(device);
    module.eval();
    this->module = module;
}
// TODO: probably may need to do error checking, also maybe casting
torch::Tensor NeRFModel::forward(std::vector<torch::jit::IValue> &inputs)
{
    // Move targets to device
    for_each(inputs.begin(), inputs.end(), [this](torch::jit::IValue &input)
             { input.toTensor().to(device); });
    torch::Tensor output = module.forward(inputs).toTensor();
    return output;
}

torch::Tensor NeRFModel::forward(torch::Tensor &inputs)
{
    auto inputsIvalue = convertToIValues(inputs);
    torch::Tensor output = module.forward(inputsIvalue).toTensor();
    return output;
}
void NeRFModel::load(std::string path)
{
    this->registerModule(torch::jit::load(path));
    this->initialized = true;
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
    for (int i = 0; i < embeddingLevel; i++)
    {
        enc.push_back(torch::sin(std::pow(2.0f, i) * x));
        enc.push_back(torch::cos(std::pow(2.0f, i) * x));
    }
    return torch::cat(enc, -1);
}

torch::Device NeRFModel::getDevice() { return device; }
bool NeRFModel::isInitialized() { return initialized; }