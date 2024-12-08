#ifndef NERFMODEL_H
#define NERFMODEL_H

#include <iostream>
#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#define slots Q_SLOTS
class NeRFModel
{
public:
    void registerModule(torch::jit::script::Module module);
    NeRFModel();
    torch::Tensor forward(std::vector<torch::jit::IValue> &inputs);
    torch::Tensor forward(torch::Tensor &inputs);
    void load(std::string path);
    torch::Device getDevice();
    bool isInitialized();
    torch::Tensor addPositionalEncoding(const torch::Tensor &x) const;

private:
    // TODO: maybe use nn module
    // jit module is used to import from pytorch
    torch::jit::script::Module module;
    torch::Device &device;
    bool initialized = false;
    int embeddingLevel = 6;
};
#endif