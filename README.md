# MonoDGC: Dynamic Graph Cross-Former for Monocular 3D Object Detection
This repository hosts the official implementation of MonoDGC: Dynamic Graph Cross-Former for Monocular 3D Object Detection based on the excellent work MonoDGP. 

### Dataset Structure
Download KITTI datasets on https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d and prepare the directory structure as:
```
│data/kitti/
├──ImageSets/
├──training/
│   ├──image_2
│   ├──label_2
│   ├──calib
├──testing/
│   ├──image_2
│   ├──calib
```
You can also change the data path at "dataset/root_dir" in configs/monodgc.yaml.
The "tools/create_data.py" script can be used to configure the format of the KITTI dataset.
```
python  tools/create_data.py kitti \
            --root-path ./data/kitti \
            --out-dir ./data/kitti \
            --extra-tag kitti
```

### Install Dependencies
```
git clone https://github.com/Tracygc/MonoDGC.git
conda create --name mmdet3d python=3.9
conda activate mmdet3d 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

For configuring a Linux environment with the RTX 5090 GPU and CUDA 12.8, you can refer to my blog post at this link:
https://blog.csdn.net/TracyGC/article/details/156197710?spm=1001.2014.3001.5501


### Get Started
#### Train
You can train MonoDG here, and you can modify the settings of models and training in configs/monodgc.yaml:
```
CUDA_VISIBLE_DEVICES=0 nohup python tools/train_val7.py --config configs/monodgc.yaml > logs/monodgc.log 2>&1
```
You can also run MonoDGP here:
```
CUDA_VISIBLE_DEVICES=0 nohup python tools/train_val.py --config configs/monodgp.yaml > logs/monodgp.log 2>&1
```
#### Test
The best checkpoint will be evaluated as default. You can change it at "tester/checkpoint" in configs/monodgc.yaml:
```
CUDA_VISIBLE_DEVICES=0 nohup python tools/train_val7.py --config configs/monodgc.yaml -e > logs/monodgc.log 2>&1
```
Test the inference time on your own device:
```
python tools/test_runtime.py
```

### Reference Links
```
https://github.com/PuFanqi23/MonoDGP
https://github.com/ZrrSkywalker/MonoDETR
https://github.com/SuperMHP/GUPNet_Plus
```



