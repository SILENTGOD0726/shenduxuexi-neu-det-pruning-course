"""Check the experiment's claims against raw records before packaging."""
import bootstrap
import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT=bootstrap.ROOT


def main():
    problems=[]
    for script in ROOT.glob('*.py'):
        ast.parse(script.read_text(encoding='utf-8'),filename=str(script))
    with (ROOT/'results/split_manifest.csv').open(encoding='utf-8') as stream:
        images=list(csv.DictReader(stream))
    assert len(images)==1800 and len({r['id'] for r in images})==1800
    assert sum(int(r['boxes']) for r in images)==4189
    hash_splits={}
    for row in images:hash_splits.setdefault(row['pixel_sha256'],set()).add(row['split'])
    assert all(len(v)==1 for v in hash_splits.values())
    total_epochs=0
    for seed in [42,43,44]:
        info=json.loads((ROOT/f'run_seed{seed}/final_info.json').read_text(encoding='utf-8'))
        assert info['matrix_complete']
        assert len(info['conditions'])==4
        for stage,epochs in [('baseline',40),('control',15),('prune20',15),('prune40',15)]:
            with (ROOT/f'run_seed{seed}/{stage}/results.csv').open(encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
            assert len(rows)==epochs,(seed,stage,len(rows))
            total_epochs+=len(rows)
        for key in ['control','prune20','prune40']:
            assert info['conditions'][key]['conversion']['allclose']
        for key in ['baseline','control','prune20_before','prune20','prune40_before','prune40']:
            result=json.loads((ROOT/f'results/seed{seed}_{key}.json').read_text(encoding='utf-8'))
            assert result['test']['split']=='test'
            assert result['test']['precision_mode']=='FP32'
            assert all(shape[-3:]==[3,320,320] for shape in result['test']['observed_input_shapes'])
            assert result['reload_check']['allclose']
            assert result['latency']['model_only_ms']['sample_count']==180
            path=ROOT/result['checkpoint']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==result['reload_check']['checkpoint_sha256']
        for key in ['prune20','prune40']:
            assert info['conditions'][key]['pruning']['after']['parameters']<info['conditions']['baseline']['validation']['parameters']
    assert total_epochs==255
    with (ROOT/'results/summary.csv').open(encoding='utf-8') as stream:summary=list(csv.DictReader(stream))
    assert len(summary)==18
    required=['accuracy_comparison.png','accuracy_cost.png','latency_comparison.png','training_curves.png',
              'annotation_examples.png','detection_cases.png','index.html']
    assert all((ROOT/'results'/name).is_file() for name in required)
    verification={'passed':True,'formal_training_epochs':total_epochs,'independent_training_seeds':[42,43,44],
                  'training_runs':12,'test_conditions':18,'image_count':len(images),'box_count':4189,
                  'no_cross_split_exact_duplicate_images':True,'all_saved_checkpoints_reload_verified':True,
                  'canonical_input_shape_verified':True,'source_syntax_checked':True}
    (ROOT/'results/verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(verification),flush=True)


if __name__=='__main__':
    main()
