import torch
import torch.nn as nn

class Net(torch.nn.Module):
    def __init__(self, net1):
        super(Net, self).__init__()
        self.net1 = net1

    def forward(self, x):
        r = torch.norm(x, p=2, dim=1, keepdim=True)
        y = self.net1(x)
        y[:, 0] += 1.0
        y[:, 4] += 1.0
        y[:, 8] += 1.0
        y = y / r
        y[(r < 0.00001).nonzero(), :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=x.device)
        return y

nn_2body = torch.load('3D_force_UB_max600_try2.pkl')
model = Net(nn_2body).cuda()

for i in range(4):
    model.to(i)
    sm = torch.jit.script(model)
    torch.jit.save(sm, "3D_force_UB_max600_try2_"+str(i)+".pt")

# trt_model = torch_tensorrt.compile(model, inputs = [torch_tensorrt.Input((128, 3), dtype=torch.float32)],
#     enabled_precisions = torch.float32
# )
# trt_model.save('3D_force_UB_max600_try2.ts')