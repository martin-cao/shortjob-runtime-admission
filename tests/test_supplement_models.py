"""Requires the real-models extra. Random CPU fixtures are not experiment data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
try:
    from torchvision import models
except ModuleNotFoundError as exc:
    if exc.name != 'torchvision':
        raise
    models = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shortjob_runner.supplement_protocol import MODELS
from shortjob_runner.supplement_models import RealImages, sha256, validate_assets
from shortjob_runner.supplement_runtime import check_trajectory


@unittest.skipIf(models is None, 'Install the real-models extra for model-specific tests')
class ModelTests(unittest.TestCase):
    def test_three_model_builders_and_pinned_weight_names(self):
        for name, weight_name in MODELS.items():
            with self.subTest(model=name):
                self.assertEqual(models.get_weight(weight_name).name, weight_name.split('.')[-1])
                model = models.get_model(name, weights=None).eval()
                with torch.inference_mode():
                    output = model(torch.zeros(1, 3, 224, 224))
                self.assertEqual(output.shape, (1, 1000))
                self.assertTrue(torch.isfinite(output).all())

    def test_prepared_inputs_cycle_identically_and_hash_tampering_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            name = 'mobilenet_v3_large'
            weights = folder / 'weights.pt'
            inputs = folder / 'inputs.pt'
            torch.save(models.get_model(name, weights=None).state_dict(), weights)
            torch.save(torch.randn(3, 3, 224, 224), inputs)
            manifest = {'models': {name: dict(weights_enum=MODELS[name], image_count=3,
                         inputs_file=inputs.name, inputs_sha256=sha256(inputs),
                         weights_file=weights.name, weights_sha256=sha256(weights))}}
            (folder / 'manifest.json').write_text(json.dumps(manifest))
            validate_assets(folder, name)
            factory = lambda w, b, d, s: RealImages(w, b, d, s, folder)
            row = check_trajectory(name, 'eager', 1, 42, [1, 2, 4], torch.device('cpu'),
                                   refresh=True, factory=factory)
            self.assertEqual(row['status'], 'passed')
            inputs.write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                validate_assets(folder, name)


if __name__ == '__main__':
    unittest.main()
