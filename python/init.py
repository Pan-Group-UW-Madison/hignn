import torch
import os
import argparse
import subprocess

if __name__ == '__main__':
    source_path = os.getcwd()
    
    os.system('clear')
    
    is_gpu_available = False
        
    try:
        subprocess.check_output('nvidia-smi')
        is_gpu_available = True
    except Exception as e:
        pass
    
    parser = argparse.ArgumentParser(description='build')
    
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the project')
    parser.add_argument('--model', type=str, default='3D_force_UB_max600_try2', help='Model name')
    
    args, unknown = parser.parse_known_args()
    
    # prepare for multi-gpu
    device_count = torch.cuda.device_count()
    
    model_name = args.model
    
    is_model_existing = os.path.exists('nn/'+model_name+'.pkl')
    if not is_model_existing:
        print('Model not found, please put the model in the nn folder')
        exit()    
    
    need_convert = False
    if is_gpu_available:
        for i in range(device_count):
            if not os.path.exists(source_path + '/nn/'+model_name+'_'+str(i)+'.pt'):
                need_convert = True
                break
    else:
        if not os.path.exists(source_path + '/nn/'+model_name+'.pt'):
            need_convert = True
    
    if need_convert:
        if is_gpu_available:
            os.system('python3 ' + './python/convert.py --gpus '+str(device_count)+' --model '+model_name)
        else:
            os.system('python3 ' + './python/convert.py --gpus 0 --model '+model_name)
    
    # build hignn library        
    if args.rebuild:
        os.system("git config --global --add safe.directory \"*\"")
        os.system("git submodule update --init --recursive")
        
        try:
            os.chdir(source_path + '/build')
        except:
            os.mkdir(source_path + '/build')
            os.chdir(source_path + '/build')
        os.system('rm -rf *')
        
        if is_gpu_available:
            os.system('export CXX=$Kokkos_PATH/bin/nvcc_wrapper && cmake -D CMAKE_PREFIX_PATH="$LibTorch_PATH/share/cmake/;$Kokkos_PATH" -D CMAKE_CXX_EXTENSIONS=Off -D USE_GPU:BOOL=On ../')
        else:
            os.system('cmake -D CMAKE_PREFIX_PATH="$LibTorch_PATH/share/cmake/" -D CMAKE_CXX_EXTENSIONS=Off -D USE_GPU:BOOL=Off ../')
            
    os.chdir(source_path + '/build')
        
    os.system('make -j 16')
    
    os.system('cp *.so ../python/hignn.so')
    
    # init running environment
    os.chdir(source_path)
    try:
        os.chdir('Result')
    except:
        os.mkdir('Result')
    # is_results_existing = os.path.exists('results')
    # if not is_results_existing:
    #     os.mkdir('results')