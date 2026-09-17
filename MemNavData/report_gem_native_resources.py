"""Report all twelve verified native-window production resource runs."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root):
    source = root / 'independent_reduction.json'
    data = json.loads(source.read_text())
    assert data['complete'] and data['verified_tasks'] == data['planned_tasks'] == 12
    assert data['plan_sha256'] == sha(root / 'plan.json')
    assert all(s['status'] == 'verified' for s in data['states'])
    for row in data['rows']:
        assert all(sha(p) == h for p,h in row['artifacts_sha256'].items())
    cases = sorted({r['task']['case'] for r in data['rows']},
        key=lambda c:next(r['frames'] for r in data['rows'] if r['task']['case'] == c))
    def values(case, window, metric):
        chosen = [r for r in data['rows'] if r['task']['case'] == case and r['task']['dense_window'] == window]
        assert {r['task']['repeat'] for r in chosen} == {0,1,2}
        if metric in ('first_ms','cached_ms'):
            queries = [next(q for q in r['queries'] if q['id'] == 'pair_00_revisit') for r in chosen]
            return np.array([q['first_ms'] if metric == 'first_ms' else q['cached_ms']['median'] for q in queries])
        return np.array([r[metric] for r in chosen])

    out = root / 'figures'
    out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.fonttype':'none'})
    metrics = ('write_seconds','gpu_peak_allocated_gib','sampled_gpu_peak_gib',
        'cpu_rss_write_final_gib','depth_archive_gib','first_ms')
    labels = ('History write time (s)','Memory-service CUDA peak (GiB)',
        'Sampled process GPU peak (GiB)','CPU RSS after history write (GiB)',
        'Historical depth archive (GiB)','First Revisit query (ms)')
    fig, axes = plt.subplots(2,3,figsize=(10.8,6.2))
    for ax, metric, label in zip(axes.flat, metrics, labels):
        for j,window in enumerate((64,16)):
            groups = [values(c,window,metric) for c in cases]
            means = np.mean(groups,axis=1)
            errors = np.array([means-np.min(groups,axis=1),np.max(groups,axis=1)-means])
            ax.bar(np.arange(2)+(j-.5)*.32,means,width=.29,yerr=errors,capsize=3,
                color=('#777777','#146c94')[j],label=f'Native W{window} + archive')
        ax.set_xticks(np.arange(2),[str(next(r['frames'] for r in data['rows'] if r['task']['case']==c)) for c in cases])
        ax.set_xlabel('Observed RGB frames');ax.set_ylabel(label)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    handles,names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,names,ncol=2,frameon=False,loc='lower center')
    fig.suptitle('Fixed production memory inputs: mean and range of 3 fresh processes',fontsize=11)
    fig.tight_layout(rect=[0,.045,1,.97])
    for suffix in ('pdf','svg','png'):
        fig.savefig(out / ('native_window_resources.'+suffix),dpi=220,bbox_inches='tight')
    plt.close(fig)

    lines = ['# 原生 W16/W64 的完整资源对照','',
        '12/12 次独立进程已完成并通过独立核验：两条固定历史 × 两种窗口 × 三次重复，按重复轮次交错窗口顺序。以下均为三次均值，图中误差线为最小值至最大值。', '',
        '| 历史帧数 | 窗口 | 全部写入，s | Torch 峰值，GiB | 进程 GPU 采样峰值，GiB | 写完 CPU RSS，GiB | 深度档案，GiB | 首次回访查询，ms | 缓存查询中位数，ms |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    summaries = []
    for case in cases:
        count = next(r['frames'] for r in data['rows'] if r['task']['case']==case)
        for window in (64,16):
            measured = {m:float(values(case,window,m).mean()) for m in (*metrics,'cached_ms')}
            summaries.append(dict(case=case,frames=count,window=window,means=measured))
            lines.append('| '+ ' | '.join([str(count),str(window),*(f'{measured[m]:.3f}' for m in (*metrics,'cached_ms'))])+' |')
    lines += ['', '同一历史内 W16 相对 W64 的写入时间与 Torch 峰值变化：','']
    for case in cases:
        count = next(r['frames'] for r in data['rows'] if r['task']['case']==case)
        changes = {m:100*(values(case,16,m).mean()/values(case,64,m).mean()-1) for m in ('write_seconds','gpu_peak_allocated_gib')}
        lines.append(f'- {count} 帧：写入 {changes["write_seconds"]:+.2f}%；Torch 峰值 {changes["gpu_peak_allocated_gib"]:+.2f}%。')
    lines += ['',
        '每次都运行真实 MemNavAgent、初始 40 帧标定和 SP/LightGlue；完整历史结束后执行一次真实首次查询及 20 次缓存查询。不是只测几何网络的轻量探针。全部写入时间包含输入读取与 SHA 核验；逐次 API 时间还单独保存在原始结果中。', '',
        'Torch 峰值在模型加载后重置，因此仍包含标定、写入和读出；进程 GPU 与 CPU 采样也覆盖加载，但可能漏掉瞬时尖峰。已有真机服务保持运行，原始采样记录同驻进程。这里没有 NavDP、Habitat 或网络调用的整机端到端时延。', '',
        'W16 仅缩小完整视图特征窗口。相机头状态、历史特殊 token、CPU 元数据和磁盘档案仍随观察数增长，不能据此声称整个模块使用恒定内存。首次目标定位与缓存目标读出具有不同工作量，应分别报告。', '',
        '这项实验只提供成本证据。W16 的长历史目标读出已出现退化；四条连续链均完成仍不足以证明完整导航与 W64 等效。70 条历史的同卡配对导航对照另行记录，不能把成本改善写成导航性能改善。', '',
        '[完整独立核验](independent_reduction.json)；[资源图 PDF](figures/native_window_resources.pdf)；[资源图 PNG](figures/native_window_resources.png)。','']
    report = root / 'RESULT.md'
    with report.open('x') as stream:stream.write('\n'.join(lines))
    receipt = dict(complete=True,source_sha256={str(source):sha(source),str(root/'plan.json'):sha(root/'plan.json')},
        script_sha256=sha(__file__),means=summaries,
        artifacts_sha256={str(p):sha(p) for p in [report,*sorted(out.iterdir())]})
    with (out/'source.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    main(parser.parse_args().root.resolve())
