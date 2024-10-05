import torch
import torch.nn as nn
import os
import argparse

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
        y[(r < 0.00001).nonzero()[:, 0].flatten(), :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=x.device)
        return y
    
if __name__ == '__main__':
    os.system('clear')
    
    parser = argparse.ArgumentParser(description='convert')
    
    parser.add_argument('--gpus', type=int, help='Number of GPUs to use', default=1)
    parser.add_argument('--model', type=str, help='Model to use')
    
    args, unknown = parser.parse_known_args()
    
    gpus = args.gpus
    model_name = args.model
    
    if model_name == None:
        print("Please provide a model name")
        exit()
    
    file_name = "./nn/"+model_name+".pkl"

    if gpus == 0:
        nn_2body = torch.load(file_name, map_location=torch.device('cpu'))
        model = Net(nn_2body)
        
        sm = torch.jit.script(model)
        target_file_name = "./nn/"+model_name+".pt"
        torch.jit.save(sm, target_file_name)
        exit()
    else:
        nn_2body = torch.load(file_name)
        model = Net(nn_2body).cuda()

        for i in range(gpus):
            model.to(i)
            sm = torch.jit.script(model)
            target_file_name = "./nn/"+model_name+"_"+str(i)+".pt"
            torch.jit.save(sm, target_file_name)