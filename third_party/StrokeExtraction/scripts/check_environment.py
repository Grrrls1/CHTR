"""Dependency smoke checks; --training also requires CUDA and training assets."""
import argparse
import importlib
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', action='store_true')
    args = parser.parse_args()
    import numpy as np
    import cv2
    import torch
    import torchvision
    import matplotlib
    matplotlib.use('Agg')
    for name in ['numpy', 'cv2', 'torch', 'torchvision', 'scipy', 'pandas', 'matplotlib', 'PIL']:
        module = importlib.import_module(name)
        print('{}: {}'.format(name, module.__version__))
    for name in ['main_train', 'train_SDNet', 'train_SegNet', 'train_ExtractNet']:
        importlib.import_module(name)
    # Exercise NumPy and OpenCV operations actually used by the project.
    sample = np.zeros((16, 16), dtype=float)
    sample[4:12, 4:12] = 1
    assert sample.astype(int).sum() == 64
    resized = cv2.resize(sample.astype(np.uint8), (32, 32))
    colored = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    cv2.putText(colored, '1', (0, 12), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255))
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / 'test.png')
        assert cv2.imwrite(path, colored)
        assert cv2.imread(path).shape == (32, 32, 3)
    x = torch.from_numpy(sample.astype(np.float32)).requires_grad_()
    x.square().sum().backward()
    assert x.grad is not None
    # torchvision compiled extension compatibility.
    torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), .5)
    print('CPU imports, NumPy/OpenCV, torch backward and torchvision NMS: PASS')
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA_VISIBLE_DEVICES: {}'.format(
        visible_devices if visible_devices is not None else '<not set>'))
    cuda_available = torch.cuda.is_available()
    print('CUDA build: {}; available: {}'.format(torch.version.cuda, cuda_available))
    cuda_compatible = False
    if cuda_available:
        device_index = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device_index)
        target = '{}{}'.format(*capability)
        arch_list = torch.cuda.get_arch_list()
        cuda_compatible = ('sm_' + target in arch_list or
                           'compute_' + target in arch_list)
        print('CUDA logical device: cuda:{}; name: {}; capability: sm_{}; '
              'wheel architectures: {}'.format(
                  device_index, torch.cuda.get_device_name(device_index), target,
                  ' '.join(arch_list)))
        if visible_devices is not None:
            visible_list = [item.strip() for item in visible_devices.split(',')]
            if device_index < len(visible_list):
                print('CUDA mapping: physical device {} -> logical cuda:{}'.format(
                    visible_list[device_index], device_index))
        print('CUDA architecture compatible: {}'.format(cuda_compatible))
    if args.training:
        missing = [str(p.relative_to(ROOT)) for p in [
            ROOT / 'char_recognise/out_vgg_bn/model/model.th',
            ROOT / 'content_net_model/out/model_content.pth',
        ] if not p.is_file()]
        for split in ['train', 'test']:
            if not any((ROOT / 'dataset/RHSEDB' / split).glob('*.npz')):
                missing.append('dataset/RHSEDB/{}/*.npz'.format(split))
        if missing:
            print('Missing training assets: ' + ', '.join(missing))
        if not cuda_available:
            print('Training requires an NVIDIA GPU visible to PyTorch.')
            return 1
        if not cuda_compatible:
            print('Training requires a PyTorch wheel that supports this GPU architecture.')
            return 1
        device = torch.device('cuda', torch.cuda.current_device())
        layer = torch.nn.Conv2d(1, 4, 3).to(device)
        layer(torch.randn(2, 1, 32, 32, device=device)).square().mean().backward()
        torch.cuda.synchronize()
        print('CUDA convolution forward/backward: PASS; ' + torch.cuda.get_device_name(0))
        return 1 if missing else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
