"""Check saved test summaries and dataset split records without model weights."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, stdev

ROOT=Path(__file__).resolve().parent

def verify():
    result_dir=ROOT/'results'
    summary=json.loads((result_dir/'summary.json').read_text(encoding='utf-8'))
    seeds=summary['seeds']
    assert seeds==[42,43,44] and summary['n']==3
    checks=0
    for condition,metrics in summary['conditions'].items():
        rows=[json.loads((result_dir/f'seed{seed}_{condition}.json').read_text(encoding='utf-8')) for seed in seeds]
        for name,record in metrics.items():
            if name in ('model_ms_median','pipeline_ms_median'):
                key='model_only_ms' if name=='model_ms_median' else 'preprocess_forward_nms_ms'
                values=[row['latency'][key]['median'] for row in rows]
            else:values=[row['test'][name] for row in rows]
            for field,value in [('mean',fmean(values)),('std',stdev(values))]:
                assert math.isclose(record[field],value,rel_tol=1e-10,abs_tol=1e-12),(condition,name,field)
                checks+=1
    with (result_dir/'split_manifest.csv').open(encoding='utf-8') as stream:images=list(csv.DictReader(stream))
    assert len(images)==1800 and len({row['id'] for row in images})==1800
    assert sum(int(row['boxes']) for row in images)==4189
    groups={}
    for row in images:groups.setdefault(row['pixel_sha256'],set()).add(row['split'])
    assert all(len(splits)==1 for splits in groups.values())
    frozen=json.loads((result_dir/'frozen_evaluation_plan.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((result_dir/'split_manifest.csv').read_bytes()).hexdigest()==frozen['manifest_sha256']
    print(f'Passed: {checks} summary values; 1800 split records; frozen split hash.')
    return checks

if __name__=='__main__':verify()
