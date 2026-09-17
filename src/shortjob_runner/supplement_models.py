"""Prepared, hashed real-image inputs and explicit pretrained weight versions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .supplement_protocol import MODELS
from .types import WorkloadSpec
from .workloads import StepWorkload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_assets(images: Path, out: Path, models, source: str, limit=64):
    from PIL import Image
    from torchvision import __version__, models as vision

    if not source.strip() or limit < 2:
        raise ValueError('A dataset source and at least two images are required')
    files = sorted(p for p in images.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and p.is_file())[:limit]
    if len(files) < 2:
        raise ValueError('Provide at least two real images; no synthetic fallback')
    if out.exists():
        raise ValueError('Asset output already exists; use a new directory to preserve provenance')
    out.mkdir(parents=True)
    manifest = {'schema': 1, 'dataset_source': source, 'torchvision': __version__,
                'images': [{'path': str(p.relative_to(images)), 'sha256': sha256(p)} for p in files], 'models': {}}
    for model_id in models:
        weights = vision.get_weight(MODELS[model_id])
        transform = weights.transforms()
        tensors = []
        for path in files:
            with Image.open(path) as image:
                tensors.append(transform(image.convert('RGB')))
        inputs_path = out / f'{model_id}.inputs.pt'
        weights_path = out / f'{model_id}.weights.pt'
        torch.save(torch.stack(tensors), inputs_path)
        # Downloads are confined to this explicit preparation command.
        model = vision.get_model(model_id, weights=weights)
        torch.save(model.state_dict(), weights_path)
        manifest['models'][model_id] = {
            'weights_enum': MODELS[model_id], 'preprocessing': repr(transform),
            'inputs_file': inputs_path.name, 'inputs_sha256': sha256(inputs_path),
            'weights_file': weights_path.name, 'weights_sha256': sha256(weights_path),
            'input_shape': list(tensors[0].shape), 'image_count': len(files),
        }
        del model
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def validate_assets(folder: Path, model_id: str) -> dict:
    data = json.loads((folder / 'manifest.json').read_text())
    item = data['models'][model_id]
    if item['weights_enum'] != MODELS[model_id] or item['image_count'] < 2:
        raise ValueError('Unexpected weights or insufficient real inputs')
    for kind in ['inputs', 'weights']:
        path = folder / item[f'{kind}_file']
        if path.resolve().parent != folder.resolve() or sha256(path) != item[f'{kind}_sha256']:
            raise ValueError(f'Asset integrity failure: {kind}')
    return item


class RealImages(StepWorkload):
    def __init__(self, model_id, batch_size, device, seed, folder: Path):
        from torchvision import models as vision

        self.spec = WorkloadSpec(workload_id=model_id, task_family='inference',
                                 workload_class='pretrained_real_images', shape_stability='stable')
        item = json.loads((folder / 'manifest.json').read_text())['models'][model_id]
        inputs = torch.load(folder / item['inputs_file'], map_location='cpu', weights_only=True)
        if batch_size > len(inputs):
            raise ValueError('batch_size exceeds prepared input count')
        # Preprocessed inputs stay on host. Every action pays the same H2D copy.
        self.batches = inputs[:len(inputs) // batch_size * batch_size].reshape(-1, batch_size, *inputs.shape[1:])
        if len(self.batches) < 2:
            raise ValueError('At least two distinct input batches are required')
        if device.type == 'cuda':
            self.batches = self.batches.pin_memory()
        self.index = seed % len(self.batches)
        self.x = self.batches[self.index].to(device)
        torch.manual_seed(seed)
        self.model = vision.get_model(model_id, weights=None)
        self.model.load_state_dict(torch.load(folder / item['weights_file'], map_location='cpu', weights_only=True))
        self.model = self.model.to(device).eval()
        self.last_output = None

    def seed_refresh(self, seed, device):
        self.index = seed % len(self.batches)

    def refresh_request_inputs(self):
        self.x.copy_(self.batches[self.index], non_blocking=True)
        self.index = (self.index + 1) % len(self.batches)
        return self.x.numel() * self.x.element_size()

    def step(self):
        self.last_output = self.model(self.x)

    def enable_compile(self, mode=None):
        self.model = torch.compile(self.model, mode=mode)

    def correctness_state(self):
        if self.last_output is None:
            raise RuntimeError('No output to inspect')
        return {'output': self.last_output.detach()}
