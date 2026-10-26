"""Single-computer follow-up: inference paths and BatchNorm recalibration.

Exploratory follow-up declared after the original test was viewed. Uses train
for recalibration and val for diagnostics; never reads test images or labels.
Original test results and checkpoints are hash-checked and left unchanged.
"""
import bootstrap
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
import warnings

import cv2
import numpy as np
import torch
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils.ops import non_max_suppression
from experiment import load_model, save_model, write_json
from training import CourseValidator, prepare_legacy_runtime

ROOT=bootstrap.ROOT
R=ROOT/'results'
WORK=ROOT/'.supplement'
SEEDS=[42,43,44]
STAGES=['baseline','control','prune20','prune40']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(image,device='cuda',half=False):
    x=cv2.cvtColor(cv2.resize(image,(320,320)),cv2.COLOR_BGR2RGB)
    x=torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).unsqueeze(0)
    return x.to(device=device,dtype=torch.float16 if half else torch.float32)/255


class ObservedValidator(CourseValidator):
    def preprocess(self,batch):
        out=super().preprocess(batch)
        self.observed_dtypes=getattr(self,'observed_dtypes',set())|{str(out['img'].dtype)}
        return out


def validate(model,name,half=False):
    args=get_cfg(overrides=dict(data=str(ROOT/'data/neu.yaml'),imgsz=320,batch=32,device='0',
        half=half,workers=0,rect=False,plots=False,save_json=False,split='val',verbose=False,
        conf=.001,iou=.7,max_det=100,task='detect'))
    validator=ObservedValidator(args=args,save_dir=WORK/name)
    validator(model=deepcopy(model))
    dtype='torch.float16' if half else 'torch.float32'
    assert validator.observed_dtypes=={dtype},validator.observed_dtypes
    assert all(s[-3:]==(3,320,320) for s in validator.observed_input_shapes)
    box=validator.metrics.box
    return {'split':'val','images':270,'mAP50':float(box.map50),'mAP50_95':float(box.map),
        'precision':float(box.mp),'recall':float(box.mr),'dtype':dtype,
        'observed_input_shapes':[list(s) for s in sorted(validator.observed_input_shapes)]}


def measure(model,images,device,half,rounds=3,repetitions=40):
    tensors=[prepare(x,device,half) for x in images]
    samples={'model_ms':[],'pipeline_ms':[]}
    sync=torch.cuda.synchronize if device=='cuda' else lambda:None
    with torch.inference_mode():
        for i in range(20):
            y=model(tensors[i%len(tensors)])[0]
            non_max_suppression(y,conf_thres=.25,iou_thres=.7,max_det=100)
        sync()
        for _ in range(rounds):
            for key in samples:
                for i in range(repetitions):
                    sync();start=time.perf_counter()
                    if key=='model_ms':model(tensors[i%len(tensors)])
                    else:
                        y=model(prepare(images[i%len(images)],device,half))[0]
                        non_max_suppression(y,conf_thres=.25,iou_thres=.7,max_det=100)
                    sync();samples[key].append((time.perf_counter()-start)*1000)
    return {k:{'median':float(np.median(v)),'p05':float(np.percentile(v,5)),
        'p95':float(np.percentile(v,95)),'sample_count':len(v),'raw_ms':v} for k,v in samples.items()}


def parameter_digest(model):
    sha=hashlib.sha256()
    for name,parameter in model.named_parameters():
        sha.update(name.encode());sha.update(parameter.detach().cpu().numpy().tobytes())
    return sha.hexdigest()


def recalibrate(model,paths):
    model.eval().float()
    before=parameter_digest(model)
    norms=[m for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
    old_momenta=[m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats();m.momentum=None;m.train()
    for p in model.parameters():p.requires_grad_(False)
    with torch.no_grad():
        for start in range(0,len(paths),32):
            batch=torch.cat([prepare(cv2.imread(str(p))) for p in paths[start:start+32]])
            model(batch)
    for m,momentum in zip(norms,old_momenta):m.momentum=momentum
    model.eval()
    unchanged=parameter_digest(model)==before
    assert unchanged
    assert all(int(m.num_batches_tracked)==20 for m in norms)
    return {'gradient_updates':0,'batchnorm_modules':len(norms),'batches':20,'images':640,
        'split':'train','trainable_parameter_values_unchanged':unchanged,
        'parameter_sha256_before_and_after':before,'policy':'Reset BN running statistics; cumulative average; only BN in train mode; other modules eval; no augmentation'}


def main():
    bootstrap.configure_torch(42);prepare_legacy_runtime()
    warnings.filterwarnings('ignore',category=FutureWarning)
    WORK.mkdir(exist_ok=True)
    infos={s:json.loads((ROOT/f'run_seed{s}/final_info.json').read_text(encoding='utf-8')) for s in SEEDS}
    frozen=[R/'summary.json',R/'summary.csv',R/'frozen_evaluation_plan.json',*R.glob('seed*.json')]
    checkpoints=[ROOT/record[k] for info in infos.values() for record in info['conditions'].values()
        for k in ['checkpoint','before_checkpoint'] if k in record]
    original={str(p.relative_to(ROOT)):digest(p) for p in frozen+checkpoints}
    train_paths=sorted((ROOT/'data/yolo/images/train').glob('*.jpg'))
    random.Random(2026).shuffle(train_paths);train_paths=train_paths[:640]
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'type':'Exploratory follow-up after original held-out results; no new held-out claims',
        'seeds':SEEDS,'timing_stages':STAGES,'timing_devices':['cuda','cpu'],'cuda_precisions':['FP32','FP16'],
        'cpu_precision':'FP32','fused':[False,True],'timing_repetitions':120,'warmup':20,
        'bn_stages':['control','prune20_before','prune40_before'],'bn_training_images':640,'bn_batch':32,
        'validation_only':True,'test_images_read':False,'original_sha256':original,
        'calibration_images':[p.name for p in train_paths],
        'sources':['https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/swa_utils.py',
            'https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/utils/fusion.py']}
    plan_path=R/'supplement_plan.json'
    if plan_path.exists():raise FileExistsError('Keep this follow-up frozen; do not silently rerun over it')
    write_json(plan_path,plan)
    images=[cv2.imread(str(p)) for p in sorted((ROOT/'data/yolo/images/val').glob('*.jpg'))[::20]]
    timing=[];fp16=[];bn=[];fusion=[]
    for seed,info in infos.items():
        for stage in STAGES:
            checkpoint=ROOT/info['conditions'][stage]['checkpoint']
            original_model=load_model(checkpoint)
            fused_model=deepcopy(original_model).fuse(verbose=False).eval().float()
            real=prepare(images[0])
            with torch.inference_mode():
                a=original_model(real)[0];b=fused_model(real)[0]
            ok=bool(torch.allclose(a,b,rtol=1e-4,atol=.002))
            if not ok:raise RuntimeError('Fused FP32 predictions changed beyond declared tolerance')
            fusion.append({'seed':seed,'stage':stage,'allclose':ok,'max_abs_error':float((a-b).abs().max()),
                'rtol':1e-4,'atol':.002,'unfused_BN_modules':sum(isinstance(m,torch.nn.BatchNorm2d) for m in original_model.modules()),
                'fused_BN_modules':sum(isinstance(m,torch.nn.BatchNorm2d) for m in fused_model.modules())})
            configs=[(device,half,fused) for device in ['cuda','cpu'] for half in ([False,True] if device=='cuda' else [False]) for fused in [False,True]]
            random.Random(seed+STAGES.index(stage)).shuffle(configs)
            for device,half,fused in configs:
                model=deepcopy(fused_model if fused else original_model).to(device)
                model=model.half() if half else model.float()
                metrics=measure(model,images,device,half)
                row={'seed':seed,'stage':stage,'device':device,'precision':'FP16' if half else 'FP32',
                     'fused':fused,'batch':1,'input':[1,3,320,320],'cpu_threads':torch.get_num_threads(),**metrics}
                timing.append(row);del model
                print('TIMING',seed,stage,device,row['precision'],fused,round(metrics['model_ms']['median'],3),flush=True)
                write_json(R/'supplement_timing.json',{'conditions':timing,'fusion_checks':fusion,
                    'scope':'320 square; batch1; 20 warmups then 3x40; synchronized CUDA; CPU4threads; pipeline excludes disk and drawing'})
            fp16.append({'seed':seed,'stage':stage,'fp32_validation':info['conditions'][stage]['validation'],
                         'fp16_validation':validate(fused_model,f'fp16_{seed}_{stage}',half=True)})
            write_json(R/'supplement_fp16_validation.json',{'conditions':fp16})
            del original_model,fused_model;torch.cuda.empty_cache()
        for stage in ['control','prune20','prune40']:
            record=info['conditions'][stage]
            checkpoint=ROOT/record['checkpoint' if stage=='control' else 'before_checkpoint']
            model=load_model(checkpoint)
            details=recalibrate(model,train_paths)
            after=validate(model,f'bn_{seed}_{stage}')
            path=WORK/f'bn_{seed}_{stage}.pt';save_model(model,path)
            before=record['validation' if stage=='control' else 'before_validation']
            bn.append({'seed':seed,'stage':stage,'before_validation':before,'after_validation':after,
                       'calibration':details,'checkpoint':str(path.relative_to(ROOT)),'checkpoint_sha256':digest(path)})
            write_json(R/'supplement_bn_validation.json',{'conditions':bn})
            print('BN_CALIBRATION',seed,stage,before['mAP50_95'],after['mAP50_95'],flush=True)
            del model;torch.cuda.empty_cache()
    assert all(digest(ROOT/p)==sha for p,sha in original.items())
    assert len(timing)==72 and len(fp16)==12 and len(bn)==9
    write_json(R/'supplement_verification.json',{'passed':True,'original_result_and_weight_hashes_unchanged':True,
        'timing_conditions':72,'samples_per_scope_per_condition':120,'FP16_validation_conditions':12,
        'BN_validation_conditions':9,'fusion_equivalence_checks':12,'all_bn_parameter_values_unchanged':True,
        'no_test_images_used':True,'status':'Exploratory validation diagnostics, not a new held-out benchmark'})
    print('SUPPLEMENT_COMPLETE',flush=True)


if __name__=='__main__':main()
