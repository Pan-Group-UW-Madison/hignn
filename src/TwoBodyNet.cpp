#include <iostream>

#include "TwoBodyNet.hpp"

void FirstNN(const int N, const int D_in, const int H, const int D_out,
             const int tstep) {
  std::cout << "*********************************************" << std::endl;
  std::cout << "Start C++" << std::endl;

  torch::manual_seed(1);
  torch::DeviceType device_type;

  if (torch::cuda::is_available() == 1) {
    std::cout << "CUDA is available! Training on GPU" << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU" << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device = device_type;

  // torch::Tensor X = torch::from_blob(foo1.data(), {N, D_in}, device_type);
  torch::Tensor X = torch::randn({N, D_in}).to(device);
  torch::Tensor Y = torch::randn({N, D_out}, device_type);

  TwoLayerNet net(D_in, H, D_out);
  net->to(device);

  torch::optim::Adam optimizer(net->parameters(),
                               torch::optim::AdamOptions(1e-4));
  torch::nn::MSELoss criterion((torch::nn::MSELossOptions(torch::kSum)));

  std::cout << "Training complete!" << std::endl;
  std::cout << "End C++" << std::endl;
  std::cout << "*********************************************" << std::endl;
  ;
}