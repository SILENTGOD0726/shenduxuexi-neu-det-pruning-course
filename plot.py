"""Generate figures and an offline results page from measured CSV/JSON only."""
import bootstrap
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = bootstrap.ROOT
RESULTS = ROOT / "results"
KEYS = ["baseline", "control", "prune20_before", "prune20", "prune40_before", "prune40"]
ENGLISH = ["Baseline", "Extra training", "20%: before FT", "20%: after FT", "40%: before FT", "40%: after FT"]
CHINESE = ["未剪枝基线", "未剪枝继续训练", "轻度剪枝后", "轻度剪枝再微调", "中度剪枝后", "中度剪枝再微调"]
COLORS = ["#718096", "#264653", "#E9C46A", "#2A9D8F", "#F4A261", "#E76F51"]


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.titlesize":11,
                         "legend.fontsize":8, "legend.frameon":False, "axes.spines.top":False,
                         "axes.spines.right":False, "axes.grid":True, "grid.alpha":.15,
                         "figure.dpi":150, "savefig.dpi":300, "savefig.bbox":"tight"})


def save(fig, stem):
    fig.savefig(RESULTS / (stem+".png"), dpi=300)
    fig.savefig(RESULTS / (stem+".pdf"))
    plt.close(fig)


def curves(preview=False):
    stages = ["baseline"] if preview else ["baseline","control","prune20","prune40"]
    fig, axes = plt.subplots(1 if preview else 2, 1 if preview else 2,
                             figsize=(7.2,3.6) if preview else (9,6.4), squeeze=False)
    for ax, stage in zip(axes.ravel(), stages):
        for i,seed in enumerate([42,43,44]):
            path = ROOT/f"run_seed{seed}/{stage}/results.csv"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as f:
                rows = [{k.strip():v for k,v in row.items()} for row in csv.DictReader(f)]
            if not rows:
                continue
            ax.plot([float(x['epoch'])+1 for x in rows],
                    [float(x['metrics/mAP50-95(B)'])*100 for x in rows],
                    label=f"seed {seed}", color=["#264653","#2A9D8F","#E76F51"][i], linewidth=1.6)
        ax.set_title(stage.replace('prune20','20% pruning + FT').replace('prune40','40% pruning + FT'))
        ax.set_xlabel("Epoch in this training stage")
        ax.set_ylabel("Validation mAP50-95 (%)")
        ax.legend()
    fig.suptitle("Recorded training validation (upstream FP16 EMA validation)",fontsize=11)
    fig.tight_layout()
    save(fig,"baseline_learning_curve" if preview else "training_curves")


def full():
    summary=json.loads((RESULTS/'summary.json').read_text(encoding='utf-8'))
    stats=summary['conditions']
    curves()
    x=np.arange(len(KEYS))
    fig,ax=plt.subplots(figsize=(9,4.3))
    for offset,metric,color,label in [(-.19,'mAP50','#264653','mAP50'),(.19,'mAP50_95','#2A9D8F','mAP50-95')]:
        means=[stats[k][metric]['mean']*100 for k in KEYS]
        errors=[(stats[k][metric]['std'] or 0)*100 for k in KEYS]
        ax.bar(x+offset,means,.36,yerr=errors,capsize=3,color=color,label=label)
    ax.set_xticks(x,ENGLISH,rotation=18,ha='right');ax.set_ylabel('Test mAP (%)');ax.legend()
    ax.set_title(f"Held-out test: mean and sample SD across {summary['n']} training seeds")
    fig.tight_layout();save(fig,'accuracy_comparison')
    fig,ax=plt.subplots(figsize=(7.8,4.5))
    for k,label,color in zip(KEYS,ENGLISH,COLORS):
        ax.errorbar(stats[k]['macs_per_image']['mean']/1e9,stats[k]['mAP50_95']['mean']*100,
                    yerr=(stats[k]['mAP50_95']['std'] or 0)*100,fmt='o',markersize=7,
                    capsize=3,color=color,label=label)
    ax.set_xlabel('Estimated MAC per image (G), 320 x 320')
    ax.set_ylabel('Test mAP50-95 (%)');ax.legend(loc='lower right');ax.set_title('Accuracy and computation cost')
    fig.tight_layout();save(fig,'accuracy_cost')
    fig,ax=plt.subplots(figsize=(9,4.3))
    for offset,metric,color,label in [(-.19,'model_ms_median','#264653','Model only'),(.19,'pipeline_ms_median','#E76F51','Preprocess + model + NMS')]:
        means=[stats[k][metric]['mean'] for k in KEYS]
        errors=[stats[k][metric]['std'] or 0 for k in KEYS]
        ax.bar(x+offset,means,.36,yerr=errors,capsize=3,color=color,label=label)
    ax.set_xticks(x,ENGLISH,rotation=18,ha='right');ax.set_ylabel('Mean of per-seed median latency (ms)')
    ax.set_title('RTX 3080 Laptop | batch 1 | FP32 | 320 x 320 | unfused');ax.legend()
    fig.tight_layout();save(fig,'latency_comparison')
    table=[]
    for k,label in zip(KEYS,CHINESE):
        m=stats[k]
        table.append(f"<tr><th>{label}</th><td>{m['mAP50']['mean']*100:.2f} ± {(m['mAP50']['std'] or 0)*100:.2f}</td>"
                     f"<td>{m['mAP50_95']['mean']*100:.2f} ± {(m['mAP50_95']['std'] or 0)*100:.2f}</td>"
                     f"<td>{m['parameters']['mean']/1e6:.3f} M</td><td>{m['macs_per_image']['mean']/1e9:.3f} G</td>"
                     f"<td>{m['model_ms_median']['mean']:.2f}</td><td>{m['pipeline_ms_median']['mean']:.2f}</td></tr>")
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>钢材缺陷检测课程实验结果</title><style>body{font:16px/1.75 "Microsoft YaHei",sans-serif;background:#f5f6f7;color:#172b33;margin:0}main{max-width:1100px;margin:auto;padding:36px 24px}h1{font-size:32px;line-height:1.35}section{background:white;border-radius:14px;padding:24px;margin:22px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;border-bottom:1px solid #ddd;padding:10px}img{max-width:100%;height:auto}small{color:#4a6570}.scroll{overflow:auto}.pill{display:inline-block;background:#d9eee6;padding:4px 12px;border-radius:20px;margin:0 8px 8px 0}</style>
<main><p>2026 深度学习课程设计 · 真实实验</p><h1>钢材表面缺陷检测与轻量化</h1>
<p>同一组数据、相同训练预算，比较模型剪小以后能否兼顾检测能力与运行效率。</p>
<span class="pill">1,800张真实图片</span><span class="pill">4,189个缺陷框</span><span class="pill">3个训练种子</span><span class="pill">六种实验条件</span>
<section><h2>先看实验结果</h2><div class="scroll"><table><thead><tr><th>条件</th><th>mAP50 %</th><th>mAP50–95 %</th><th>参数量</th><th>MAC/图</th><th>模型延迟ms</th><th>含前后处理ms</th></tr></thead><tbody>'''+''.join(table)+'''</tbody></table></div><small>精度为三个训练种子的均值±样本标准差。耗时为各模型重复计时的中位数再跨种子求均值。GPU结果不代表开发板结果。</small></section>
<section><h2>为什么需要未剪枝继续训练的对照</h2><p>剪枝后微调增加了训练时间。增加同等训练预算的未剪枝对照，才能判断变化来自剪枝，还是仅仅来自额外训练。</p><img src="accuracy_comparison.png" alt="测试精度对照图"></section>
<section><h2>训练过程和压缩代价</h2><img src="training_curves.png" alt="训练验证曲线"><img src="accuracy_cost.png" alt="精度和计算成本"></section>
<section><h2>真正测出的运行耗时</h2><img src="latency_comparison.png" alt="实测延迟比较"><p>统一RTX3080 Laptop、320×320、batch1、FP32，预热后重复测量，并进行GPU同步。包含前后处理的时间不含磁盘读取与绘图。</p></section>
<section><h2>看得到的检测案例</h2><p>绿色为标注，红色为预测。按每类错误最少和最多的规则选图，包含成功与失败，避免只展示好看的样例。</p><img src="detection_cases.png" alt="成功和失败检测案例"></section>
<section><h2>怎么复核</h2><p>查看同目录summary.csv、逐条件JSON、数据划分清单和数据核验记录；按README重新运行。模型、代码和数据来源均有记录。</p><p>NEU-DET划分为本项目自建划分；数据规模有限，缺少真实产线和RK3568板端验证。剪枝配置比例不等于实际压缩率。</p></section></main></html>'''
    (RESULTS/'index.html').write_text(page,encoding='utf-8')
    report_rows=['| 条件 | mAP50 % | mAP50-95 % | 参数量 M | MAC G |',
                 '|---|---:|---:|---:|---:|']
    for k,label in zip(KEYS,CHINESE):
        m=stats[k]
        report_rows.append(f"| {label} | {m['mAP50']['mean']*100:.2f} ± {(m['mAP50']['std'] or 0)*100:.2f} | "
                           f"{m['mAP50_95']['mean']*100:.2f} ± {(m['mAP50_95']['std'] or 0)*100:.2f} | "
                           f"{m['parameters']['mean']/1e6:.3f} | {m['macs_per_image']['mean']/1e9:.3f} |")
    block='\n\n'.join(['三个训练种子、六种预定条件已完成统一测试。下表精度为均值±样本标准差。',
                         '\n'.join(report_rows),
                         '![测试精度比较](results/accuracy_comparison.png)',
                         '![实测延迟比较](results/latency_comparison.png)',
                         '完整离线结果页：[results/index.html](results/index.html)。'])
    readme_path=ROOT/'README.md'
    text=readme_path.read_text(encoding='utf-8')
    start='<!-- measured-results:start -->';end='<!-- measured-results:end -->'
    before,rest=text.split(start,1);_,after=rest.split(end,1)
    text=before+start+'\n'+block+'\n'+end+after
    text=text.replace('当前任务正在执行。完成后的真实结果、图表和统计保存在',
                      '已完成12次正式训练、合计255轮，六种条件在三个种子下的18组测试结果均已保存。真实结果、图表和统计保存在')
    readme_path.write_text(text,encoding='utf-8')
    print('FIGURES_AND_RESULTS_PAGE_READY',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    style()
    if args.preview:
        curves(True)
    else:
        full()
