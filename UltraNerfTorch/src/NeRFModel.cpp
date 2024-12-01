#include "NeRFModel.h"

NeRFModel::NeRFModel(torch::jit::script::Module module)
{
    module.to(device);
    module.eval();
    this->module = module;
}
// TODO: probably may need to do error checking, also maybe casting
NeRFModel::NeRFModel() {}
torch::Tensor NeRFModel::infer(std::vector<torch::jit::IValue> inputs)
{
    // Move targets to device
    for_each(inputs.begin(), inputs.end(), [this](torch::jit::IValue &input)
             { input.toTensor().to(device); });
    torch::Tensor output = module.forward(inputs).toTensor();
    return output;
}

NeRFModel NeRFModel::load(std::string path, bool toCuda)
{
    torch::jit::script::Module module;
    if (toCuda)
    {
        module = torch::jit::load(path, torch::kCUDA);
    }
    else
    {
        module = torch::jit::load(path, torch::kCPU);
        device = torch::Device(torch::kCPU);
    }
    return NeRFModel(module);
}

torch::Device NeRFModel::getDevice() { return device; }