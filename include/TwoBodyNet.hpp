#ifndef _TwoBodyNet_Hpp_
#define _TwoBodyNet_Hpp_

#include <torch/torch.h>

#include "Typedef.hpp"

struct TwoLayerNetImpl : torch::nn::Module {
  TwoLayerNetImpl(int D_in, int H, int D_out)
      : linear1(torch::nn::LinearOptions(D_in, H).bias(false)),
        linear2(torch::nn::LinearOptions(H, D_out).bias(false)) {
    register_module("linear1", linear1);
    register_module("linear2", linear2);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::clamp_min(linear1(x), 0);
    x = linear2(x);
    return x;
  }
  torch::nn::Linear linear1, linear2;
};
TORCH_MODULE(TwoLayerNet);

void FirstNN(const int N, const int D_in, const int H, const int D_out,
             const int tstep);

#endif