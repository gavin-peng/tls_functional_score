"""
Regenerate paper/figures/spatial_tls_scores.png as a proper tissue-section figure.

For 3 representative samples:
  - all spots plotted as tiny gray dots (tissue context)
  - TLS cluster spots coloured by GNN tls_functional_score (blue=immunogenic, red=tolerogenic)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
H5AD  = ROOT / 'data/processed/rcc_visium_labeled.h5ad'
PREDS = ROOT / 'data/tls_predictions.csv'
OUT   = ROOT / 'paper/figures/spatial_tls_scores.png'

SAMPLES = [
    'GSM5924031_ffpe_c_3',   # 240 TLS, good spread
    'GSM5924030_ffpe_c_2',   # 150 TLS, highest mean score
    'GSM5924033_ffpe_c_7',   # 281 TLS, largest sample
]
LABELS = ['c_3', 'c_2', 'c_7']

def main():
    print('Loading adata (obs + obsm only)...')
    adata = sc.read_h5ad(H5AD, backed='r')
    obs = adata.obs[['sample_id', 'tls_cluster_id']].copy()
    spatial = pd.DataFrame(
        adata.obsm['spatial'],
        index=adata.obs.index,
        columns=['x', 'y']
    )
    obs = obs.join(spatial)

    print('Loading predictions...')
    preds = pd.read_csv(PREDS)[['cluster_id', 'sample_id', 'tls_functional_score']]

    cmap = plt.cm.RdBu_r

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, sid, label in zip(axes, SAMPLES, LABELS):
        mask = obs['sample_id'] == sid
        df = obs[mask].copy()

        # background: all spots
        ax.scatter(df['x'], df['y'], c='#d0d0d0', s=0.3, linewidths=0, rasterized=True)

        # TLS spots: merge with preds on cluster_id
        tls_mask = df['tls_cluster_id'].notna() & (df['tls_cluster_id'] >= 0)
        df_tls = df[tls_mask].copy()
        df_tls['tls_cluster_id'] = df_tls['tls_cluster_id'].astype(int)

        preds_sid = preds[preds['sample_id'] == sid]
        df_tls = df_tls.merge(
            preds_sid[['cluster_id', 'tls_functional_score']],
            left_on='tls_cluster_id', right_on='cluster_id', how='left'
        )

        sc_ = ax.scatter(
            df_tls['x'], df_tls['y'],
            c=df_tls['tls_functional_score'],
            cmap=cmap, vmin=0, vmax=1,
            s=4, linewidths=0, rasterized=True
        )

        # flip y axis to match tissue orientation (Visium y increases downward)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(label, fontsize=10, fontweight='bold')

        n_tls_spots = tls_mask.sum()
        n_clusters = df_tls['cluster_id'].nunique()
        ax.text(0.02, 0.02, f'{n_clusters} TLS clusters\n{n_tls_spots:,} spots',
                transform=ax.transAxes, fontsize=7, color='#444444',
                va='bottom')

    # shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('GNN score\n(0=immunogenic, 1=tolerogenic)', fontsize=8)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['0\n(immunogenic)', '0.5', '1\n(tolerogenic)'])
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        'Spatial distribution of TLS functional scores in RCC tissue sections',
        fontsize=10, y=1.01
    )
    plt.subplots_adjust(left=0.01, right=0.91, top=0.93, bottom=0.02, wspace=0.04)
    fig.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    main()
