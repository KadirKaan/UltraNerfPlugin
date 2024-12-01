#ifndef NERFMODEL_H
#define NERFMODEL_H

#include <iostream>
#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#define slots Q_SLOTS
class NeRFModel
{
    torch::jit::Module module;
    torch::Device device = torch::Device(torch::kCUDA);

public:
    NeRFModel(torch::jit::script::Module module);
    NeRFModel();
    torch::Tensor infer(std::vector<torch::jit::IValue> inputs);
    NeRFModel load(std::string path, bool toCuda = true);
    torch::Device getDevice();
};
#endif