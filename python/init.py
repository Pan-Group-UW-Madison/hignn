import torch
import os
import argparse

if __name__ == '__main__':
    source_path = os.getcwd()
    
    parser = argparse.ArgumentParser(description='build')
    
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the project')
    parser.add_argument('--model', type=str, default='3D_force_UB_max600_try2', help='Model name')
    
    args, unknown = parser.parse_known_args()
    
    # prepare for multi-gpu
    device_count = torch.cuda.device_count()
    
    os.system('clear')
    
    model_name = args.model
    
    is_model_existing = os.path.exists('nn/'+model_name+'.pkl')
    if not is_model_existing:
        print('Model not found, please put the model in the nn folder')
        exit()    
    
    need_convert = False
    for i in range(device_count):
        if not os.path.exists(source_path + '/nn/'+model_name+'_'+str(i)+'.pt'):
            need_convert = True
            break
    
    if need_convert:
        os.system('python3 ' + './python/convert.py --gpus '+str(device_count)+' --model '+model_name)
    
    # build hignn library
    os.chdir(source_path + '/build')
    if args.rebuild:
        os.system('rm -rf *')
        
        os.system('export CXX=$Kokkos_PATH/bin/nvcc_wrapper && cmake -D CMAKE_PREFIX_PATH="$LibTorch_PATH/share/cmake/;$Kokkos_PATH" -D CMAKE_CXX_EXTENSIONS=Off ../')
        
    os.system('make -j 16')
    
    os.system('cp *.so ../python/hignn.so')
    
    # # init running environment
    # os.chdir(source_path)
    # is_results_existing = os.path.exists('results')
    # if not is_results_existing:
    #     os.mkdir('results')