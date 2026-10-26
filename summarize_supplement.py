"""Summarize and plot the measured single-computer diagnostics."""
import bootstrap
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=bootstrap.ROOT
R=ROOT/'results'
KEYS=['control','prune20','prune40']
LABELS=['Extra training','20% pruning + FT','40% pruning + FT']


def read(name):return json.loads((R/name).read_text(encoding='utf-8'))


def stats(values):
    return {'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1)),'n':len(values)}


def main():
    assert read('supplement_verification.json')['passed'] and read('onnx_verification.json')['passed']
    timing=read('supplement_timing.json');bn=read('supplement_bn_validation.json')['conditions']
    precision=read('supplement_fp16_validation.json')['conditions'];onnx=read('onnx_validation.json')['conditions']
    runtime={}
    for key in KEYS:
        runtime[key]={}
        for device,half,fused,name in [('cuda',False,False,'gpu_unfused_fp32'),('cuda',False,True,'gpu_fused_fp32'),
                ('cuda',True,True,'gpu_fused_fp16'),('cpu',False,False,'cpu_unfused_fp32'),('cpu',False,True,'cpu_fused_fp32')]:
            rows=[r for r in timing['conditions'] if r['stage']==key and r['device']==device and r['precision']==('FP16' if half else 'FP32') and r['fused']==fused]
            assert len(rows)==3
            runtime[key][name]={m:stats([r[m]['median'] for r in rows]) for m in ['model_ms','pipeline_ms']}
        rows=[r for r in onnx if r['stage']==key];assert len(rows)==3
        runtime[key]['onnx_cpu_fp32']={m:stats([r['latency'][m]['median'] for r in rows]) for m in ['model_ms','pipeline_ms']}
    recovery={}
    for key in KEYS:
        rows=[r for r in bn if r['stage']==key]
        recovery[key]={'before':stats([r['before_validation']['mAP50_95'] for r in rows]),
            'recalibrated':stats([r['after_validation']['mAP50_95'] for r in rows]),
            'finetuned':stats([json.loads((ROOT/f'run_seed{s}/final_info.json').read_text(encoding='utf-8'))['conditions'][key]['validation']['mAP50_95'] for s in [42,43,44]])}
    result={'n':3,'runtime':runtime,'bn_recovery_validation':recovery,
        'max_fp16_validation_AP_difference':max(abs(r['fp16_validation']['mAP50_95']-r['fp32_validation']['mAP50_95']) for r in precision),
        'max_onnx_validation_AP_difference':max(abs(r['validation_AP_delta']) for r in onnx),
        'max_onnx_output_absolute_error':max(r['parity_max_abs_error'] for r in onnx),
        'max_fp32_fusion_output_absolute_error':max(r['max_abs_error'] for r in timing['fusion_checks']),
        'scope':'Follow-up validation diagnostics and local runtime measurement; original held-out results unchanged'}
    (R/'supplement_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.dpi':250,'savefig.bbox':'tight'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));x=np.arange(3)
    for ax,paths,title in [(axes[0],[('gpu_unfused_fp32','FP32 unfused'),('gpu_fused_fp32','FP32 fused'),('gpu_fused_fp16','FP16 fused')],'RTX 3080 Laptop GPU'),
            (axes[1],[('cpu_unfused_fp32','PyTorch unfused'),('cpu_fused_fp32','PyTorch fused'),('onnx_cpu_fp32','ONNX Runtime')],'CPU, FP32, 4 threads')]:
        for i,(path,label) in enumerate(paths):
            values=[runtime[k][path]['model_ms'] for k in KEYS]
            ax.bar(x+(i-1)*.25,[r['mean'] for r in values],.24,yerr=[r['std'] for r in values],
                   capsize=3,label=label,color=['#718096','#264653','#2A9D8F'][i])
        ax.set_xticks(x,LABELS,rotation=15,ha='right');ax.set_ylabel('Model latency (ms)');ax.set_title(title);ax.legend(fontsize=8)
    fig.suptitle('Batch 1, 320 x 320 | mean of per-seed timing medians (n=3)');fig.tight_layout()
    fig.savefig(R/'runtime_paths.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4));x=np.arange(2)
    for i,(name,label,color) in enumerate([('before','Immediately pruned','#E9C46A'),('recalibrated','BN recalibration only','#718096'),('finetuned','15 epochs fine-tuning','#2A9D8F')]):
        values=[recovery[k][name] for k in ['prune20','prune40']]
        ax.bar(x+(i-1)*.25,[v['mean']*100 for v in values],.24,yerr=[v['std']*100 for v in values],capsize=3,label=label,color=color)
    ax.set_xticks(x,['20% channel setting','40% channel setting']);ax.set_ylabel('Validation mAP50-95 (%)')
    ax.set_title('BN uses 640 train images and zero gradient updates');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(R/'bn_recovery.png');plt.close(fig)
    cn={'control':'未剪枝继续训练','prune20':'轻度剪枝后微调','prune40':'中度剪枝后微调'}
    rows=['| 条件 | GPU未融合 | GPU融合 | GPU融合FP16 | CPU融合 | CPU ONNX |','|---|---:|---:|---:|---:|---:|']
    for k in KEYS:
        rows.append('| '+cn[k]+' | '+' | '.join(f"{runtime[k][p]['model_ms']['mean']:.2f}" for p in ['gpu_unfused_fp32','gpu_fused_fp32','gpu_fused_fp16','cpu_fused_fp32','onnx_cpu_fp32'])+' |')
    b=[f"| {cn[k]} | {recovery[k]['before']['mean']*100:.2f} | {recovery[k]['recalibrated']['mean']*100:.2f} | {recovery[k]['finetuned']['mean']*100:.2f} |" for k in ['prune20','prune40']]
    document='''# 单机补充验证

这轮补充实验在查看原始测试结果后设计，属于探索性分析。归一化校准只读取训练图片，精度核对使用270张验证图。原18组测试结果与训练权重保持原样，补充结果没有冒充新一轮独立测试。

## 同一模型换一种运行方式

三个训练种子各测未融合、卷积/归一化融合和不同精度路径，共72组配置。每组预热20次、三轮各测40次，保存原始120次记录与P05/P95。CUDA计时前后同步；CPU固定4个算子线程。模型计时不含预处理、NMS、读盘与绘图，完整流水线另有记录。下表单位为毫秒，除特别标注均为FP32，是每模型中位数再跨种子求均值。

'''+ '\n'.join(rows)+'''

![不同运行路径](../results/runtime_paths.png)

融合前后检查了12个模型的实际图片输出，均通过预先规定的容差。融合减少了运行时需要执行的归一化模块；其耗时变化有直接测量支持。FP16并未在当前单图、小模型路径中带来速度收益，不能把较低精度自动当成较快推理。这些结果仍受本机、线程、输入和后端限制。

## 归一化统计是否导致剪枝失效

每个模型只用固定640张训练图片重新计算BatchNorm均值与方差，20个batch，关闭增强，不反向传播。参数值的SHA256在前后完全一致，改变的是归一化统计。未剪枝继续训练模型也做相同处理，保留这一对照。

| 剪枝条件 | 立即剪枝 | 只重算BN | 15轮微调 |
|---|---:|---:|---:|
'''+ '\n'.join(b)+'''

表中为验证集mAP50-95百分数，三个种子的均值。重算统计能恢复一部分精度，仍远低于微调结果。因此当前证据不支持“只修正统计就能替代微调”。也不能把剪枝后的全部损失归结为单一原因。

![统计校准与微调](../results/bn_recovery.png)

## CPU独立运行时

另导出9个静态ONNX模型：三个种子乘三个已微调条件。输入固定为batch1、3通道、320×320，FP32，opset12。ONNX检查器完整检查通过；每个模型用14张真实验证图片核对原始输出，并在全部270张验证图上计算检测指标。这里复用相同的检测指标计算，网络推理由ONNX Runtime的CPUExecutionProvider完成，未调用GPU推理。

运行版本为ONNX1.17.0、ONNX Runtime1.20.1；算子线程4、算子间线程1，启用图优化。包及其PyPI哈希记录在onnx_package_manifest.json。额外依赖位于.onnx_deps，没有替换原训练环境。该验证证明本机CPU导出路径可用，不能代替另一台电脑或开发板验收。

## 来源和复核入口

- [PyTorch2.5.1的Conv/BN融合实现](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/utils/fusion.py)：读取后使用固定检测源码的fuse入口并核对输出。
- [PyTorch2.5.1的归一化统计更新参考](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/swa_utils.py)：本项目只把BN置为训练模式，保留检测头的评估行为；未运行SWA参数平均。
- [ONNX Runtime Python接口](https://onnxruntime.ai/docs/api/python/api_summary.html)和[线程设置](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)。

原始文件：supplement_plan.json、supplement_timing.json、supplement_bn_validation.json、supplement_fp16_validation.json、onnx_validation_plan.json、onnx_validation.json及两份verification。配置在运行前保存，失败和不改善结果均保留。
'''
    control_note=f"未剪枝对照重算统计后，验证mAP50-95由{recovery['control']['before']['mean']*100:.2f}%变为{recovery['control']['recalibrated']['mean']*100:.2f}%。这项处理也不保证改善原模型。\n\n"
    document=document.replace('## CPU独立运行时',control_note+'## CPU独立运行时')
    document=document.replace('CPU固定4个算子线程','CPU为Intel Core i7-11800H（8核16线程），固定4个算子线程')
    (ROOT/'docs/单机补充验证.md').write_text(document,encoding='utf-8')
    block='\n'.join(['<!-- single-pc-followup:start -->','## 单机补充验证','',
        '已完成72组推理路径测速、12组FP16验证、9组BN统计校准和9个CPU ONNX模型的导出验证。以下是同机模型延迟（毫秒），三个训练种子对应中位数的均值。','',*rows,'',
        '![本机运行路径](results/runtime_paths.png)','',
        '[详细条件、未改善结果与来源](docs/单机补充验证.md)。这些是追加的验证集诊断，原始测试结果保持不变。',
        '<!-- single-pc-followup:end -->'])
    path=ROOT/'README.md';text=path.read_text(encoding='utf-8')
    if '<!-- single-pc-followup:start -->' in text:text=re.sub(r'<!-- single-pc-followup:start -->.*?<!-- single-pc-followup:end -->',block,text,flags=re.S)
    else:text+='\n\n'+block+'\n'
    path.write_text(text,encoding='utf-8')
    if (R/'index.html').exists():
        path=R/'index.html';text=path.read_text(encoding='utf-8')
        text=re.sub(r'<section id="single-pc-followup">.*?</section>','',text,flags=re.S)
        text=text.replace('</main>', '<section id="single-pc-followup"><h2>单机补充验证</h2><p>追加验证仅用训练和验证数据，保留原始测试。已验证融合、FP16、BN重算与CPU ONNX路径。</p><img src="runtime_paths.png" alt="本机运行路径"><img src="bn_recovery.png" alt="BN统计校准"></section></main>')
        path.write_text(text,encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
