from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _get(name, default):
    return os.environ.get(name, default)

@dataclass(frozen=True)
class Settings:
    api_key: str = _get('FORTE_API_KEY', '')
    detector_weights: str = _get('FORTE_DETECTOR_WEIGHTS', '')
    action_checkpoint: str = _get('FORTE_ACTION_CHECKPOINT', '')
    action_backend: str = _get('FORTE_ACTION_BACKEND', 'auto')
    device: str = _get('FORTE_DEVICE', '0')
    work_dir: Path = Path(_get('FORTE_WORK_DIR', str(ROOT / 'storage'))).resolve()
    sample_fps: float = float(_get('FORTE_SAMPLE_FPS', '2'))
    detector_imgsz: int = int(_get('FORTE_DETECTOR_IMGSZ', '768'))
    detector_conf: float = float(_get('FORTE_DETECTOR_CONF', '0.35'))
    action_threshold: float = float(_get('FORTE_ACTION_THRESHOLD', '0.65'))
    action_window_sec: float = float(_get('FORTE_ACTION_WINDOW_SEC', '4'))
    action_stride_sec: float = float(_get('FORTE_ACTION_STRIDE_SEC', '2'))
    max_duration_sec: float = float(_get('FORTE_MAX_DURATION_SEC', '180'))
    max_upload_mb: int = int(_get('FORTE_MAX_UPLOAD_MB', '200'))
    job_timeout_sec: int = int(_get('FORTE_JOB_TIMEOUT_SEC', '285'))
    rules_path: Path = Path(_get('FORTE_RULES_PATH', str(ROOT / 'config' / 'fortebank_procedure_rules.json'))).resolve()
    store_internal_details: bool = _get('FORTE_SAVE_INTERNAL', '1') == '1'

    def assert_ready(self):
        if len(self.api_key) < 12:
            raise ValueError('FORTE_API_KEY must be set and have at least 12 characters')
        # Accepts a .pt file, an exported .onnx file, or an OpenVINO *_openvino_model directory.
        if not self.detector_weights or not Path(self.detector_weights).exists():
            raise FileNotFoundError('Set FORTE_DETECTOR_WEIGHTS to the detector best.pt / .onnx / _openvino_model')
        if self.action_checkpoint and not Path(self.action_checkpoint).is_file():
            raise FileNotFoundError(f'FORTE_ACTION_CHECKPOINT not found: {self.action_checkpoint}')
        if self.sample_fps <= 0 or self.action_stride_sec <= 0:
            raise ValueError('Sampling FPS and action stride must be positive')
