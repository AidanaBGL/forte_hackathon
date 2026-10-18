"""Persistent single-instance SQLite + Railway Volume queue.
Only 1 Railway replica should be deployed when using this SQLite implementation.
"""
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from contextlib import contextmanager


class Store:
    def __init__(self, directory: Path):
        self.root = directory
        self.uploads = directory / 'uploads'
        self.results = directory / 'results'
        self.db_file = directory / 'jobs.sqlite3'

    def init(self):
        for d in (self.root, self.uploads, self.results):
            d.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(d, 0o700)
            except OSError:
                pass
        with self.db() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, stage TEXT NOT NULL,
                filename TEXT NOT NULL, upload_path TEXT NOT NULL,
                sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                created_at REAL NOT NULL, started_at REAL, finished_at REAL,
                updated_at REAL NOT NULL, lease TEXT, worker_id TEXT,
                error TEXT, verdict TEXT, process_name TEXT, duration_sec REAL,
                result_path TEXT, video_retained INTEGER NOT NULL DEFAULT 0
            )''')
            c.execute('CREATE INDEX IF NOT EXISTS idx_jobs_queue ON jobs(status,created_at)')
            c.execute('''CREATE TABLE IF NOT EXISTS workers (
                worker_id TEXT PRIMARY KEY, last_seen REAL NOT NULL,
                detector_classes TEXT NOT NULL, action_classes TEXT NOT NULL
            )''')

    @contextmanager
    def db(self):
        c=sqlite3.connect(str(self.db_file),timeout=10,isolation_level=None)
        c.row_factory=sqlite3.Row
        try:
            c.execute('PRAGMA busy_timeout=10000')
            c.execute('PRAGMA journal_mode=WAL')
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    def presence(self, worker_id, detector_classes, action_classes):
        with self.db() as c:
            c.execute("""INSERT INTO workers (worker_id,last_seen,detector_classes,action_classes)
                     VALUES (?,?,?,?) ON CONFLICT(worker_id) DO UPDATE SET
                     last_seen=excluded.last_seen,detector_classes=excluded.detector_classes,
                     action_classes=excluded.action_classes""",
                      (worker_id[:80],time.time(),json.dumps(detector_classes),json.dumps(action_classes)))

    def workers(self):
        with self.db() as c:
            return [dict(r) for r in c.execute('SELECT * FROM workers ORDER BY last_seen DESC LIMIT 8').fetchall()]

    def count_pending(self):
        with self.db() as c:
            return int(c.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','processing')").fetchone()[0])

    def create(self, name, path, sha256, size):
        now=time.time()
        job=uuid.uuid4().hex[:16]
        with self.db() as c:
            c.execute('INSERT INTO jobs (id,status,stage,filename,upload_path,sha256,size_bytes,created_at,updated_at) '
                      'VALUES (?,?,?,?,?,?,?,?,?)', (job,'queued','В очереди',name,str(path),sha256,size,now,now))
        return job

    def one(self, job_id):
        with self.db() as c:
            r=c.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
            return dict(r) if r else None

    def all(self, limit=50):
        with self.db() as c:
            return [dict(r) for r in c.execute('SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?', (limit,)).fetchall()]

    def expire(self, max_seconds):
        """Mark jobs exceeding the competition wall-clock deadline as failed."""
        threshold=time.time()-max_seconds
        with self.db() as c:
            c.execute("UPDATE jobs SET status='failed',stage='Истёк лимит времени',"
                      "error='Analysis did not finish within the configured deadline',updated_at=?,finished_at=? "
                      "WHERE status IN ('queued','processing') AND created_at < ?",
                      (time.time(),time.time(),threshold))

    def claim(self, worker_id):
        with self.db() as c:
            c.execute('BEGIN IMMEDIATE')
            row=c.execute("SELECT id,filename,size_bytes FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if not row:
                c.execute('COMMIT')
                return None
            lease=uuid.uuid4().hex
            now=time.time()
            c.execute("UPDATE jobs SET status='processing',stage='Обработка на GPU',started_at=?,updated_at=?,"
                      "lease=?,worker_id=? WHERE id=?",(now,now,lease,worker_id,row['id']))
            c.execute('COMMIT')
            return dict(row)|{'lease':lease}

    def is_lease(self, job_id, lease):
        r=self.one(job_id)
        return r and r['status']=='processing' and r['lease']==lease

    def stage(self, job_id, lease, value):
        with self.db() as c:
            cur=c.execute("UPDATE jobs SET stage=?,updated_at=? WHERE id=? AND lease=? AND status='processing'",
                          (str(value)[:160],time.time(),job_id,lease))
            return cur.rowcount==1

    def complete(self, job_id, lease, result, details, save_video):
        r=self.one(job_id)
        if not r or r['status']!='processing' or r['lease']!=lease:
            return False
        payload={'job_id':job_id,'status':'done','verdict':result['verdict'],
                 'violations':result['violations'],'details':details}
        dest=self.results/f'{job_id}.json'
        tmp=dest.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
        os.replace(tmp,dest)
        with self.db() as c:
            cur=c.execute('''UPDATE jobs SET status='done',stage='Готово',finished_at=?,updated_at=?,
                  verdict=?,process_name=?,duration_sec=?,result_path=?,video_retained=?
                  WHERE id=? AND lease=? AND status='processing' ''',
                  (time.time(),time.time(),result['verdict'],str(details.get('process',{}).get('name','')),
                   details.get('processing_sec'),str(dest),int(save_video),job_id,lease))
        if cur.rowcount!=1:
            dest.unlink(missing_ok=True)
            return False
        if not save_video:
            Path(r['upload_path']).unlink(missing_ok=True)
        return True

    def fail(self, job_id, lease, error, save_video):
        r=self.one(job_id)
        with self.db() as c:
            cur=c.execute("UPDATE jobs SET status='failed',stage='Ошибка',error=?,finished_at=?,updated_at=?,"
                          "video_retained=? WHERE id=? AND lease=? AND status='processing'",
                          (str(error)[:800],time.time(),time.time(),int(save_video),job_id,lease))
        if cur.rowcount and not save_video and r:
            Path(r['upload_path']).unlink(missing_ok=True)
        return cur.rowcount==1

    def delete(self, job_id):
        r=self.one(job_id)
        if not r: return False
        if r['status'] in ('queued','processing'):
            raise ValueError('Cannot delete a queued or processing job')
        with self.db() as c:
            c.execute('DELETE FROM jobs WHERE id=?',(job_id,))
        if r.get('result_path'): Path(r['result_path']).unlink(missing_ok=True)
        Path(r['upload_path']).unlink(missing_ok=True)
        return True

    def load_result(self, row):
        p=row.get('result_path')
        if not p or not Path(p).is_file():return None
        return json.loads(Path(p).read_text(encoding='utf-8'))
