"""Standalone figures from verified GEM evidence; no manuscript edits."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS={'native':'#555555','camera':'#b57521','reciprocal':'#146c94'}
LABELS={'native':'Native interval-7','camera':'Connected (OLS)','reciprocal':'Connected (reciprocal)'}


def finish(fig, out, name):
    for suffix in ('pdf','svg','png'):
        fig.savefig(out/(name+'.'+suffix),dpi=220,bbox_inches='tight')
    plt.close(fig)


def geometry(root,out):
    source=root/'readout_001/independent_verification.json'
    data=json.loads(source.read_text())
    assert data['verified'] and data['pose_to_bearing_verified']
    rows=data['rows']
    fig,axes=plt.subplots(1,3,figsize=(11,3.1),sharey=True)
    scenes=('LT9Jq6dN3Ea','Qpor2mEya8F','5jp3fCRSRjc')
    for ax,scene in zip(axes,scenes):
        selected=sorted([r for r in rows if r['case'].startswith(scene) and r['role']=='revisit'],key=lambda r:r['prefix'])
        for mode in COLORS:
            x=[r['prefix'] for r in selected]
            y=[r['bearing_deg'][mode] if r['bearing_deg'][mode] is not None else np.nan for r in selected]
            ax.plot(x,y,'o-',color=COLORS[mode],lw=1.5,ms=3.5,label=LABELS[mode])
        ax.set_title(scene)
        ax.set_xlabel('Observed RGB frames')
        ax.grid(alpha=.18)
        ax.set_ylim(0,180)
    axes[0].set_ylabel('Goal bearing error (degrees)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,frameon=False,loc='lower center',bbox_to_anchor=(.5,-.07))
    fig.suptitle('Long-history development queries: all preselected prefixes',fontsize=11)
    fig.tight_layout()
    finish(fig,out,'long_history_goal_bearing')

    selected=[r for r in rows if r['role']=='revisit' and r['complete_history']]
    accepted=[r for r in selected if all(r['bearing_deg'][m] is not None for m in ('native','reciprocal'))]
    fig,ax=plt.subplots(figsize=(5.4,4.4))
    ax.plot([.01,180],[.01,180],'--',lw=1,color='#999999')
    for row in accepted:
        x,y=row['bearing_deg']['native'],row['bearing_deg']['reciprocal']
        ax.scatter(x,y,color=COLORS['reciprocal'],s=35)
        label=row['case'].split('_episode')[0][:4]
        ax.annotate(label,(x,y),xytext=(5,4),textcoords='offset points',fontsize=9)
    ax.set(xscale='log',yscale='log',xlim=(.02,200),ylim=(.02,200),
        xlabel='Native interval-7 goal error (degrees)',ylabel='Connected reciprocal goal error (degrees)')
    ax.grid(alpha=.18,which='both')
    ax.set_title(f'Complete-history Revisit: {len(accepted)} jointly accepted / {len(selected)} queries',fontsize=10)
    fig.text(.5,.005,'Development diagnostics; lower than diagonal indicates smaller error.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.035,1,1])
    finish(fig,out,'complete_history_revisit_bearing')
    return source


def costs(root,out):
    source=root/'resources_001/independent_reduction.json'
    data=json.loads(source.read_text())
    assert data['complete'],'Do not present incomplete cost repetitions as final'
    modes=('legacy','native_interval7','connected_reciprocal')
    names=('Legacy + replay','Native + archive','Connected + archive')
    cases=sorted({r['task']['case'] for r in data['rows']},key=lambda c:next(r['frames'] for r in data['rows'] if r['task']['case']==c))
    fig,axes=plt.subplots(2,3,figsize=(11,6.4))
    metrics=('write_seconds','gpu_peak_allocated_gib','sampled_gpu_peak_gib',
        'cpu_rss_write_final_gib','depth_archive_gib','revisit_first_ms')
    labels=('Total history write time (s)','Memory-service CUDA peak (GiB)',
        'Sampled process GPU peak (GiB)','CPU RSS after history write (GiB)',
        'Historical depth archive (GiB)','First Revisit goal query (ms; log scale)')
    def value(row,metric):
        if metric=='revisit_first_ms':
            return next(q['first_ms'] for q in row['queries'] if q['id']=='pair_00_revisit')
        return row[metric]
    for ax,metric,label in zip(axes.flat,metrics,labels):
        for j,mode in enumerate(modes):
            records=[[value(r,metric) for r in data['rows'] if r['task']['case']==case and r['task']['mode']==mode] for case in cases]
            assert all(len(r)==3 for r in records)
            means=np.mean(records,axis=1)
            errors=np.array([means-np.min(records,axis=1),np.max(records,axis=1)-means])
            ax.bar(np.arange(len(cases))+(j-1)*.25,means,width=.23,yerr=errors,capsize=3,
                label=names[j],color=('#777777','#b57521','#146c94')[j])
        ax.set_xticks(np.arange(len(cases)),[str(next(r['frames'] for r in data['rows'] if r['task']['case']==c)) for c in cases])
        ax.set_xlabel('Observed RGB frames')
        ax.set_ylabel(label)
        if metric=='revisit_first_ms':
            ax.set_yscale('log')
            ax.set_ylim(50,50000)
        ax.grid(axis='y',alpha=.18)
        ax.set_axisbelow(True)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,frameon=False,loc='lower center',bbox_to_anchor=(.5,-.012))
    fig.suptitle('Fixed-input production memory: mean and range over 3 fresh processes',fontsize=11)
    fig.tight_layout(rect=[0,.035,1,1])
    finish(fig,out,'production_memory_cost')
    return source


def navigation(root,out):
    source=root/'hpc_003/final/independent_reduction.json'
    data=json.loads(source.read_text())
    assert data['complete'] and data['verified_tasks']==210
    assert data['expected_rollouts']==420 and data['paired_histories']==70
    groups=data['groups']
    datasets=('all','hm3d','mp3d')
    modes=('legacy','native_interval7','connected_reciprocal')
    names=('Current GEM','Native + archive','Connected + archive')
    colors=('#777777','#b57521','#146c94')
    fig,axes=plt.subplots(2,2,figsize=(9,6.4))
    for row,role in enumerate(('novel','revisit')):
        for column,metric in enumerate(('sr','spl')):
            ax=axes[row,column]
            for j,mode in enumerate(modes):
                values=[groups[f'{d}/all_scenes/{role}']['arms'][mode][metric] for d in datasets]
                ax.bar(np.arange(3)+(j-1)*.25,values,width=.23,label=names[j],color=colors[j])
            counts=[groups[f'{d}/all_scenes/{role}']['arms']['legacy']['n'] for d in datasets]
            ax.set_xticks(np.arange(3),[f'{d.upper() if d!="all" else "All"}\n(n={n})' for d,n in zip(datasets,counts)])
            ax.set_ylim(0,1.05)
            ax.set_ylabel('Success rate' if metric=='sr' else 'SPL')
            ax.set_title(role.capitalize())
            ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,frameon=False,loc='lower center',bbox_to_anchor=(.5,-.005))
    fig.suptitle('Frozen 70-history comparison: all 420 navigation rollouts verified',fontsize=11)
    fig.tight_layout(rect=[0,.045,1,1])
    finish(fig,out,'navigation_success_spl')

    fig,axes=plt.subplots(2,2,figsize=(10,6.7))
    comparisons=[(d,c) for d in datasets for c in ('legacy','native_interval7')]
    row_labels=[f'{d.upper() if d!="all" else "All"}: vs {"current GEM" if c=="legacy" else "native"}' for d,c in comparisons]
    for row,role in enumerate(('novel','revisit')):
        for column,metric in enumerate(('reached','spl')):
            ax=axes[row,column]
            factor=100 if metric=='reached' else 1
            for j,(dataset,control) in enumerate(comparisons):
                difference=groups[f'{dataset}/all_scenes/{role}']['paired_differences'][control][metric]
                low,high=difference['scene_bootstrap_ci95']
                color=colors[0 if control=='legacy' else 1]
                ax.hlines(j,low*factor,high*factor,color=color,lw=2)
                ax.plot(difference['mean']*factor,j,'o',color=color,ms=4)
            ax.axvline(0,color='#999999',lw=1,ls='--')
            ax.set_yticks(range(len(comparisons)),row_labels)
            ax.invert_yaxis()
            ax.set_xlabel('Connected success difference (percentage points)' if metric=='reached' else 'Connected SPL difference')
            ax.set_title(role.capitalize())
            ax.grid(axis='x',alpha=.18)
    fig.suptitle('Paired differences with scene-cluster bootstrap intervals',fontsize=11)
    fig.text(.5,.004,'10,000 fixed-seed draws; descriptive 95% intervals, not a prespecified noninferiority test.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.035,1,1])
    finish(fig,out,'navigation_paired_differences')
    return source


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('geometry','costs','navigation'))
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    out=args.root/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    source={'geometry':geometry,'costs':costs,'navigation':navigation}[args.phase](args.root,out)
    receipt=dict(source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),paper_modified=False)
    (out/(args.phase+'_source.json')).write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
