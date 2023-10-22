# DsworksEqualAI
Dsworks AIJ 2023 Equal AI Competition
1. Датасет содержит 1000 классов жестов РЖЯ по 20 видео в каждом классе = 20000 видео
2. В разметке есть начало и конец жеста - обрежем видео для удаления ненужных кадров и уменьшения размера датасета
3. Переводим текст в метку класса
4. делим датасет на train, val, с условием, что человек из train не принадлежит val

Install cuda win10
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

Cuda support:

pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html

Info env:

python -m torch.utils.collect_env

import torch; print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.backends.cudnn.enabled)


conda install cuda -c nvidia/label/cuda-12.1.0

nvcc -V

Install MMAction2 locally with cuda in conda environment:

0. conda create --name openmmlab python=3.8 -y 
  conda activate openmmlab
1. conda install cuda -c nvidia/label/cuda-12.1.0
2. nvcc -V
3. conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
4. python -m torch.utils.collect_env
5. import torch; print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.backends.cudnn.enabled)
6. conda install -c anaconda git
7. git –-version
8. pip install -U openmim
9. mim install mmengine
10. mim install mmcv
11. git clone https://github.com/open-mmlab/mmaction2.git
    
cd mmaction2

pip install -v -e .

13. mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest .
    
14.  python demo/demo.py tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth  demo/demo.mp4 tools/data/kinetics/label_map_k400.txt