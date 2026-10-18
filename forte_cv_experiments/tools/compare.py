"""Read results from independent experiments and print comparable heldout metrics."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');a=p.parse_args()
print(f'{"run":28} {"recall":>8} {"fpr":>8} {"f1":>8} {"time(s)":>9}')
for run in a.runs:
    d=json.loads((Path(run)/'metrics.json').read_text())
    m=d['test']
    print(f'{Path(run).name[:28]:28} {m["violation_recall"]:8.2f} {m["false_positive_rate"]:8.2f} {m["f1"]:8.2f} {d["run_seconds"]:9.1f}')
