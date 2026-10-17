"""Replay the unchanged protocol in a temporary copy and compare archives.

This verifies reproducibility, not a new model-selection attempt. The original
frozen/final directories are never removed, modified, or overwritten.
"""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,tempfile
from datetime import datetime,timezone
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
checks=[]
with tempfile.TemporaryDirectory(prefix='safeinvest-reproduce-') as folder:
    work=Path(folder)
    for directory in ['src','data']:
        shutil.copytree(ROOT/directory,work/directory,ignore=shutil.ignore_patterns('__pycache__'))
    (work/'docs').mkdir();shutil.copy2(ROOT/'docs/protocol.json',work/'docs/protocol.json')
    (work/'scripts').mkdir();shutil.copy2(ROOT/'scripts/run_study.py',work/'scripts/run_study.py')
    for stage in ['develop','freeze','final']:
        process=subprocess.run([sys.executable,str(work/'scripts/run_study.py'),stage],capture_output=True,text=True)
        if process.returncode:raise RuntimeError(f'{stage} failed: {process.stderr[-2000:]}')
        print('Reproduced',stage)
    for relative in ['development/summary.csv','final/summary.csv','final/cost_sensitivity.csv',
                     'final/horizon15.csv','final/onboarding_diagnostics.csv','frozen/onboarding.csv']:
        original=pd.read_csv(ROOT/'results'/relative)
        replay=pd.read_csv(work/'results'/relative)
        pd.testing.assert_frame_equal(original,replay,check_exact=False,rtol=1e-12,atol=1e-12)
        checks.append(relative)
    for relative in ['development/paths.npz','final/paths.npz']:
        with np.load(ROOT/'results'/relative) as a,np.load(work/'results'/relative) as b:
            assert set(a.files)==set(b.files)
            for key in a.files:np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-12)
        checks.append(relative)
    original=json.loads((ROOT/'results/frozen/manifest.json').read_text())
    replay=json.loads((work/'results/frozen/manifest.json').read_text())
    for key in ['selected','portfolios','assignments','fingerprints','forecast_sha256']:
        assert original[key]==replay[key]
        checks.append('frozen.'+key)
record={'status':'PASS','completed_utc':datetime.now(timezone.utc).isoformat(),
        'comparisons':checks,'original_archives_modified':False,
        'parameters_or_selection_rule_changed':False,
        'scope':'Exact same-protocol reproduction in disposable directory; no new tuning or policy choice.'}
(ROOT/'results/reproduction.json').write_text(json.dumps(record,indent=2))
print('PASS:',len(checks),'archive and model comparisons; original evidence retained.')
