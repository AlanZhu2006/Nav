"""Show both continuous regressions, without changing measured trajectories."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root):
    data=json.loads((root/'continuous_diagnosis.json').read_text())
    assert data['complete'] and data['actual_legs']==35
    modes=('legacy','native_interval7','connected_reciprocal')
    labels=('Current GEM','Native64 + archive','Connected memory')
    colors=('#2878B5','#5B9A56','#C44E52')
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(10,6.6),layout='constrained')
    for index,(scene,leg,title) in enumerate((('pLe4wQe7qrG',1,'pLe: historical goal B'),
                                             ('yqstnuAEVhm',2,'yqst: returned goal A'))):
        path_ax,error_ax=axes[index]
        for mode,label,color in zip(modes,labels,colors):
            row=next(r for r in data['rows'] if r['scene']==scene and r['leg']==leg and r['mode']==mode)
            folder=root/f"tasks/{row['task']}/evaluation/leg_{leg}"
            trace=json.loads((folder/'actual_trace.json').read_text())
            path=np.array([[p['x'],p['z']] for p in trace['poses']]+[trace['end_position'][::2]])
            path_ax.plot(path[:,0],path[:,1],color=color,lw=1.6,label=f'{label} ({"success" if row["reached"] else "failure"})')
            path_ax.scatter(*path[0],color=color,marker='s',s=22,zorder=4)
            path_ax.scatter(*path[-1],color=color,marker='o' if row['reached'] else 'x',s=32,zorder=4)
            plans=[p for p in row['plans'] if p['bearing_error_deg'] is not None]
            error_ax.plot([p['step'] for p in plans],[p['bearing_error_deg'] for p in plans],
                color=color,lw=1.5,label=label)
        goal=row['goal_xz']
        path_ax.add_patch(Circle(goal,1.,facecolor='#EAC85E',edgecolor='#B28A13',alpha=.17))
        path_ax.scatter(*goal,color='#B28A13',marker='*',s=85,zorder=4,label='Goal (1 m success region)')
        path_ax.set(xlabel='World x (m)',ylabel='World z (m)',title=title)
        path_ax.set_aspect('equal',adjustable='datalim')
        path_ax.legend(fontsize=7,loc='best',framealpha=.9)
        error_ax.set(xlabel='Executed actions within this leg',ylabel='Goal-bearing error (degrees)',
            ylim=(-2,182),title='Error along each method’s own executed path')
        error_ax.legend(fontsize=7,loc='upper left')
        for ax in (path_ax,error_ax): ax.grid(alpha=.17)
    output=root/'figures'
    output.mkdir(exist_ok=True)
    files=[]
    for suffix in ('pdf','svg','png'):
        path=output/f'continuous_regressions.{suffix}'
        fig.savefig(path,dpi=180)
        files.append(path)
    plt.close(fig)
    (output/'receipt.json').write_text(json.dumps(dict(
        source_sha256=sha(root/'continuous_diagnosis.json'),plotter_sha256=sha(__file__),
        files={p.name:sha(p) for p in files},
        scope='Both observed continuous failures; posthoc trajectories differ across arms; squares are actual leg starts; no wall geometry rendered'),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    run(parser.parse_args().root.resolve())
