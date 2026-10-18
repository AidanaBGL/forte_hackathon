from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    api_key: str = os.getenv('FORTE_API_KEY', '')
    worker_key: str = os.getenv('FORTE_WORKER_KEY', '')
    storage_dir: Path = Path(os.getenv('FORTE_STORAGE_DIR', '/data')).resolve()
    save_videos: bool = os.getenv('FORTE_SAVE_VIDEOS', '0').lower() in {'1', 'true', 'yes'}
    max_upload_mb: int = int(os.getenv('FORTE_MAX_UPLOAD_MB', '150'))
    max_pending: int = int(os.getenv('FORTE_MAX_PENDING', '10'))
    job_timeout_sec: int = int(os.getenv('FORTE_JOB_TIMEOUT_SEC', '295'))
    allow_mp4: bool = os.getenv('FORTE_ALLOW_MP4', '1') == '1'

    def validate(self):
        if len(self.api_key) < 16:
            raise RuntimeError('FORTE_API_KEY is required (at least 16 characters)')
        if len(self.worker_key) < 16 or self.api_key == self.worker_key:
            raise RuntimeError('FORTE_WORKER_KEY must differ from FORTE_API_KEY (at least 16 characters)')
        if self.max_upload_mb < 1 or self.max_pending < 1 or self.job_timeout_sec < 1:
            raise RuntimeError('Upload size, queue and timeout must be positive')
