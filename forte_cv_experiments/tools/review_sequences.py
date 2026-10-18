import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.rules import run
p=argparse.ArgumentParser()
p.add_argument('--actions-dir',default='runs/action_predictions')
p.add_argument('--out',default='runs/sequence_checks')
a=p.parse_args()
run(a.actions_dir,a.out)
