"""Export nine fixed models and verify CPU runtime outputs, val AP and latency."""
import onnx_bootstrap
import bootstrap
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import time
import warnings

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics.nn.modules import Detect
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils.ops import non_max_suppression
from experiment import load_model,write_json
from supplement import ObservedValidator
from training import prepare_legacy_runtime
from onnx_runtime import prepare,session,OrtAdapter

ROOT=bootstrap.ROOT
R=ROOT/'results'
OUT=ROOT/'.supplement/onnx'


def run_validation(runtime,names,tag):
    args=get_cfg(overrides=dict(data=str(ROOT/'data/neu.yaml'),imgsz=320,batch=1,device='cpu',
        half=False,workers=0,rect=False,plots=False,save_json=False,split='val',verbose=False,
        conf=.001,iou=.7,max_det=100,task='detect'))
    validator=ObservedValidator(args=args,save_dir=OUT/tag)
    validator(model=OrtAdapter(runtime,names))
    assert validator.observed_dtypes=={'torch.float32'}
    assert validator.observed_input_shapes=={(1,3,320,320)}
    return {'split':'val','images':270,'mAP50':float(validator.metrics.box.map50),
            'mAP50_95':float(validator.metrics.box.map),'batch':1,'runtime':'ONNX Runtime CPU',
            'precision':'FP32','observed_input_shape':[1,3,320,320]}


def benchmark(runtime,images):
    tensors=[prepare(im) for im in images]
    samples={'model_ms':[],'pipeline_ms':[]}
    for x in tensors[:1]*20:runtime.run(['predictions'],{'images':x})
    for _ in range(3):
        for key in samples:
            for i in range(40):
                started=time.perf_counter()
                if key=='model_ms':runtime.run(['predictions'],{'images':tensors[i%len(tensors)]})
                else:
                    y=runtime.run(['predictions'],{'images':prepare(images[i%len(images)])})[0]
                    non_max_suppression(torch.from_numpy(y),conf_thres=.25,iou_thres=.7,max_det=100)
                samples[key].append((time.perf_counter()-started)*1000)
    return {k:{'median':float(np.median(v)),'p05':float(np.percentile(v,5)),
        'p95':float(np.percentile(v,95)),'sample_count':len(v),'raw_ms':v} for k,v in samples.items()}


def main():
    bootstrap.configure_torch(42);prepare_legacy_runtime()
    warnings.filterwarnings('ignore',category=FutureWarning)
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'seeds':[42,43,44],
        'stages':['control','prune20','prune40'],'shape':[1,3,320,320],'opset':12,
        'exporter':'torch2.5.1 legacy TorchScript exporter; dynamo=False; static batch1',
        'runtime':ort.__version__,'onnx':onnx.__version__,'provider':'CPUExecutionProvider',
        'intra_threads':4,'inter_threads':1,'parity_rtol':.0001,'parity_atol':.002,
        'validation_AP_maximum_absolute_tolerance':.005,'validation_only':True,
        'test_images_read':False,'timing_warmup':20,'timing_measurements':120,
        'source':'https://onnxruntime.ai/docs/api/python/api_summary.html'}
    plan_path=R/'onnx_validation_plan.json'
    if plan_path.exists():raise FileExistsError('Keep the original deployment plan and partial results')
    write_json(plan_path,plan)
    images=[cv2.imread(str(p)) for p in sorted((ROOT/'data/yolo/images/val').glob('*.jpg'))[::20]]
    rows=[]
    for seed in plan['seeds']:
        info=json.loads((ROOT/f'run_seed{seed}/final_info.json').read_text(encoding='utf-8'))
        for stage in plan['stages']:
            record=info['conditions'][stage]
            checkpoint=ROOT/record['checkpoint'];weight_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            model=load_model(checkpoint).cpu().float().eval().fuse(verbose=False)
            exporter=deepcopy(model)
            for m in exporter.modules():
                if isinstance(m,Detect):m.export=True;m.format='onnx';m.dynamic=False
            sample=torch.zeros(1,3,320,320)
            with torch.no_grad():exporter(sample)
            target=OUT/f'seed{seed}_{stage}.onnx'
            torch.onnx.export(exporter,sample,str(target),input_names=['images'],output_names=['predictions'],
                opset_version=12,dynamo=False,do_constant_folding=True)
            graph=onnx.load(str(target));onnx.checker.check_model(graph,full_check=True)
            for key,value in {'stride':'32','names':str(model.names),'course_input':'1,3,320,320'}.items():
                item=graph.metadata_props.add();item.key=key;item.value=value
            onnx.save(graph,str(target));runtime=session(target)
            errors=[];passed=[]
            with torch.inference_mode():
                for image in images:
                    x=prepare(image);reference=model(torch.from_numpy(x))[0].numpy()
                    actual=runtime.run(['predictions'],{'images':x})[0]
                    assert actual.shape==(1,10,2100)
                    errors.append(float(np.max(np.abs(actual-reference))))
                    passed.append(bool(np.allclose(actual,reference,rtol=.0001,atol=.002)))
            if not all(passed):raise RuntimeError(f'ONNX output parity failed: {seed} {stage} {max(errors)}')
            validation=run_validation(runtime,model.names,f'val_{seed}_{stage}')
            difference=validation['mAP50_95']-record['validation']['mAP50_95']
            assert abs(difference)<=.005,(seed,stage,difference)
            timings=benchmark(runtime,images)
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==weight_sha
            rows.append({'seed':seed,'stage':stage,'checkpoint':record['checkpoint'],'checkpoint_sha256':weight_sha,
                'onnx_path':str(target.relative_to(ROOT)),'onnx_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                'onnx_bytes':target.stat().st_size,'onnx_checker_passed':True,'parity_images':len(images),
                'parity_allclose':True,'parity_max_abs_error':max(errors),'parity_rtol':.0001,'parity_atol':.002,
                'validation':validation,'original_GPU_batch32_val':record['validation']['mAP50_95'],
                'validation_AP_delta':difference,'latency':timings,'providers':runtime.get_providers()})
            write_json(R/'onnx_validation.json',{'plan':plan,'conditions':rows})
            print('ONNX_VERIFIED',seed,stage,max(errors),difference,timings['model_ms']['median'],flush=True)
            del model,exporter,runtime;torch.cuda.empty_cache()
    assert len(rows)==9
    write_json(R/'onnx_verification.json',{'passed':True,'exports':9,'full_validation_images_each':270,
        'parity_images_each':len(images),'runtime':'ONNX Runtime CPU','no_test_images_used':True,
        'original_checkpoints_unchanged':True,'all_validation_AP_differences_within_predeclared_tolerance':True})
    print('ONNX_COMPLETE',flush=True)


if __name__=='__main__':main()
