"""Inspect dependencies, tensor interoperability, optional CUDA and dataset."""
import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--require-torchvision', action='store_true')
    parser.add_argument('--device', default='cuda:0', help='Logical CUDA device for the kernel check')
    parser.add_argument('--dataset', help='Optional LMDB directory; validates mapping and first sample')
    parser.add_argument('--output-json', help='Optional diagnostic report file')
    args = parser.parse_args()
    report = {'python': sys.version.split()[0], 'executable': sys.executable,
              'platform': platform.platform(), 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'packages': {}, 'errors': [], 'warnings': []}
    modules = {}
    for name in ('torch', 'numpy', 'cv2', 'lmdb', 'torchvision'):
        try:
            modules[name] = importlib.import_module(name)
            report['packages'][name] = getattr(modules[name], '__version__', 'unknown')
        except Exception as exc:
            message = '%s import failed: %s' % (name, exc)
            if name == 'torchvision' and not args.require_torchvision:
                report['warnings'].append(message + ' (only needed for resnet18)')
            else:
                report['errors'].append(message)
    if 'torch' in modules:
        torch = modules['torch']
        try:
            x = torch.ones(4, requires_grad=True)
            x.square().sum().backward()
            if 'numpy' in modules:
                a = modules['numpy'].ones(2, dtype='float32')
                torch.from_numpy(a).numpy()
            report['cpu_tensor_check'] = 'passed'
        except Exception as exc:
            report['errors'].append('CPU/NumPy tensor check failed: ' + str(exc))
        cuda = {'build_version': torch.version.cuda, 'available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count()}
        report['cuda'] = cuda
        if cuda['available']:
            try:
                device = torch.device(args.device)
                if device.type != 'cuda':
                    raise ValueError('--device must be a CUDA device')
                properties = torch.cuda.get_device_properties(device)
                cuda.update({'selected_device': str(device), 'name': properties.name,
                             'memory_gib': round(properties.total_memory/1024**3, 2),
                             'capability': list(torch.cuda.get_device_capability(device)),
                             'compiled_arches': torch.cuda.get_arch_list()})
                x = torch.randn(32, 32, device=device, requires_grad=True)
                (x @ x.T).square().mean().backward()
                torch.cuda.synchronize(device)
                cuda['kernel_and_backward_check'] = 'passed'
            except Exception as exc:
                report['errors'].append('CUDA kernel check failed: ' + str(exc))
        elif args.require_cuda:
            report['errors'].append('CUDA unavailable: check NVIDIA driver, GPU visibility and CUDA-enabled PyTorch')
        else:
            report['warnings'].append('CUDA unavailable; CPU checks only')
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                                    capture_output=True, text=True, timeout=10)
            report['nvidia_smi'] = {'returncode': result.returncode,
                                    'output': (result.stdout or result.stderr).strip()}
        except Exception as exc:
            report['warnings'].append('nvidia-smi failed: ' + str(exc))
    if args.dataset:
        dataset = None
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from models.datasets.lmdb_dataset import CHTRModelDataset
            dataset = CHTRModelDataset(args.dataset)
            sample = dataset[0]
            report['dataset'] = {'path': dataset.path, 'samples': len(dataset),
                                 'type_map': 'stroke_type_map.json', 'first_sample': sample['sample_id'],
                                 'image_shape': list(sample['image'].shape),
                                 'stroke_count': len(sample['stroke_types'])}
        except Exception as exc:
            report['errors'].append('dataset check failed: ' + str(exc))
        finally:
            if dataset is not None:
                dataset.close()
    report['ok'] = not report['errors']
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered+'\n', encoding='utf-8')
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
