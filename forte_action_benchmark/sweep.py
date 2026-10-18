"""Run a reproducible experiment grid, rank on VALIDATION only."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='experiments.json')
    ap.add_argument('--manifest',default='runs/data')
    ap.add_argument('--output',default='runs/experiments')
    ap.add_argument('--limit',type=int,default=0,help='Run only first N experiments (0=all)')
    ap.add_argument('--test-best',action='store_true',help='Evaluate ONE validation-selected model on held-out test')
    ap.add_argument('--allow-cpu',action='store_true',help='Explicitly permit slow CPU training')
    args=ap.parse_args()
    import torch
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError('CUDA not available. Check nvidia-smi and install matching '
                           'torch/torchvision CUDA build. Use --allow-cpu only intentionally.')
    experiments=json.loads(Path(args.config).read_text(encoding='utf-8'))['experiments']
    if args.limit:experiments=experiments[:args.limit]
    results=[]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for exp in experiments:
        name=exp['name']
        kwargs={k:v for k,v in exp.items() if k!='name'}
        cmd=[sys.executable,'train.py','train','--manifest',args.manifest,'--out',str(out/name)]
        if args.allow_cpu and not torch.cuda.is_available(): cmd.append('--cpu')
        for k,v in kwargs.items():
            flag='--'+k.replace('_','-')
            if isinstance(v,bool):
                cmd.append(flag if v else '--no-'+k.replace('_','-'))
            elif v is not None:
                cmd += [flag,str(v)]
        print('\n========',name,'========',flush=True)
        try:
            subprocess.run(cmd,check=True)
            m=json.loads((out/name/'val_metrics.json').read_text())
            row={'name':name,'checkpoint':str(out/name/'best.pt'),
                 'model':exp['model'],'loss':exp['loss'],'aug':exp['aug'],**m,'status':'ok'}
        except subprocess.CalledProcessError as e:
            row={'name':name,'model':exp.get('model'),'status':f'failed rc={e.returncode}'}
        results.append(row)
        pd.DataFrame(results).to_csv(out/'leaderboard.csv',index=False)
    successes=[r for r in results if r['status']=='ok']
    if not successes:
        raise RuntimeError('Every experiment failed. Inspect terminal logs and dataset.')
    best=sorted(successes,key=lambda x:(x['macro_f1'],x['accuracy']),reverse=True)[0]
    print('\nVALIDATION WINNER:',best['name'], 'macroF1:',best['macro_f1'])
    print('LEADERBOARD:',out/'leaderboard.csv')
    if args.test_best:
        subprocess.run([sys.executable,'train.py','test','--manifest',args.manifest,
                        '--checkpoint',best['checkpoint'],'--out',str(out/'heldout_test')],check=True)
        print('TEST results:',out/'heldout_test')
    else:
        print('Test untouched. When happy with VAL model, use --test-best ONCE.')

if __name__=='__main__':main()
