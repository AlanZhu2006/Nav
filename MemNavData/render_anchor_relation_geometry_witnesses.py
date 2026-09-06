#!/usr/bin/env python3
"""Render measured geometry/masks for inspection; not a generated method figure."""
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    args=p.parse_args()
    out=args.root/'geometry_witnesses.png'
    if out.exists():raise FileExistsError(out)
    sys.path.insert(0,'/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map')
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    witnesses=sorted(args.root.glob('witness_*.npz'))
    fig,axes=plt.subplots(len(witnesses),3,figsize=(11,3.6*len(witnesses)),squeeze=False)
    for row,path in enumerate(witnesses):
        with np.load(path) as data:
            rgb_path=data['rgb_path'].item();depth=data['depth'];mask=data['content_mask']
            image=load_and_preprocess_images([rgb_path],mode='pad',image_size=518,patch_size=14)[0]
        a=axes[row]
        a[0].imshow(image.permute(1,2,0).numpy());a[0].set_title(f'Historical RGB, frame {path.stem[8:]}')
        valid=depth[mask & np.isfinite(depth) & (depth>0)]
        shown=np.where(mask,depth,np.nan)
        a[1].imshow(shown,cmap='gray_r',vmin=np.quantile(valid,.02),vmax=np.quantile(valid,.98))
        a[1].set_title('Causal LingBot depth (relative units)')
        key=f'17DRP5sb8fy__episode_0000.npz'
        with np.load(args.root/key) as g:
            fraction=g[f'{path.stem[8:]}/valid_fraction'].reshape(8,8)
        a[2].imshow(fraction,cmap='gray',vmin=0,vmax=1)
        a[2].set_title('Aligned 8 x 8 valid fraction')
        for col in a:col.set_axis_off()
    fig.tight_layout();fig.savefig(out,dpi=160);plt.close(fig)
    print(out)


if __name__=='__main__':main()
