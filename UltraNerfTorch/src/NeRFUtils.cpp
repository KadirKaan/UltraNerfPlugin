#include "NeRFUtils.h"

#include <fstream>
#include <iostream>
torch::Device getDevice()
{
    return torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
}