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
    NeRFModel(const torch::Device &device, int D = 8, int W = 256, int embeddingLevel = 6);
    torch::Tensor forward(torch::Tensor &inputs);
    void load(std::string path);
    torch::Device getDevice();
    bool isInitialized();
    torch::Tensor addPositionalEncoding(const torch::Tensor &x) const;

private:
    // TODO: maybe use nn module
    torch::nn::Sequential model_;
    const torch::Device &device_;
    bool initialized_ = false;
    int embeddingLevel_ = 6;
};
#endif