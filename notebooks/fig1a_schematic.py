"""Figure 1A — study overview schematic (matplotlib draft)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).parent.parent / 'checkpoints' / 'figure1a_schematic.png'

fig, ax = plt.subplots(figsize=(17, 9))
ax.set_xlim(0, 17)
ax.set_ylim(0, 9)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── palette ───────────────────────────────────────────────────────────────
CD  = '#dbeafe'; BD = '#1d4ed8'   # data   — blue
CM  = '#dcfce7'; BM = '#15803d'   # method — green
CR  = '#fef3c7'; BR = '#b45309'   # result — amber

def box(x, y, w, h, fc, ec, lw=1.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.08',
                                fc=fc, ec=ec, lw=lw, zorder=2))

def arr(x1, y1, x2, y2, rad=0.0, color='#555555', lw=1.6, ls='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle=ls,
                                connectionstyle=f'arc3,rad={rad}'))

def t(x, y, s, **kw):
    ax.text(x, y, s, ha='center', va='top', **kw)

# ── section headers ────────────────────────────────────────────────────────
t(2.1,  8.8, 'DATA',        fontsize=13, fontweight='bold', color=BD)
t(7.2,  8.8, 'METHODS',     fontsize=13, fontweight='bold', color=BM)
t(13.5, 8.8, 'KEY RESULTS', fontsize=13, fontweight='bold', color=BR)

ax.plot([4.1, 4.1], [0.3, 8.7], color='#d1d5db', lw=1, ls='--', zorder=0)
ax.plot([10.3, 10.3], [0.3, 8.7], color='#d1d5db', lw=1, ls='--', zorder=0)

# ── DATA ──────────────────────────────────────────────────────────────────
# RCC
box(0.2, 5.0, 3.7, 3.3, CD, BD)
t(2.05, 8.0,  'RCC Visium',             fontsize=10, fontweight='bold', color=BD)
t(2.05, 7.55, '73,280 spots',           fontsize=9)
t(2.05, 7.15, 'GSE175540 / TCGA-KIRC',  fontsize=8,  color='#555')
t(2.05, 6.65, 'Pseudo-labels (post-hoc only):',
                                         fontsize=7.5, color='#777', style='italic')
t(2.05, 6.25, 'TLS  ·  NO_TLS',        fontsize=9,  fontweight='bold', color=BD)
t(2.05, 5.75, '(used only to name clusters\nafter unsupervised fitting)',
                                         fontsize=7.5, color='#999', style='italic',
                                         linespacing=1.4)

# KC/LC
box(0.2, 0.9, 3.7, 3.8, CD, BD)
t(2.05, 4.4,  'KC/LC Visium',                    fontsize=10, fontweight='bold', color=BD)
t(2.05, 3.95, '19,404 spots  ·  6 samples',      fontsize=9)
t(2.05, 3.55, 'Nonchev et al. 2025',             fontsize=8,  color='#555')
t(2.05, 3.05, 'Pathologist annotations:',        fontsize=7.5, color='#777', style='italic')
t(2.05, 2.6,  'ETLS  ·  MTLS  ·  TUM  ·  NOR',  fontsize=9,  fontweight='bold', color=BD)
t(2.05, 2.1,  'Held-out evaluation only —\nnever seen during clustering',
                                                   fontsize=7.5, color='#999', style='italic',
                                                   linespacing=1.4)
t(2.05, 1.35, 'Excl: KC2 (RNA quality), LC4 (misalign),\nLC3 LN spots (33 spots)',
                                                   fontsize=7,  color='#bbb',
                                                   linespacing=1.4)

# ── METHODS ───────────────────────────────────────────────────────────────
# Step 1 — signature scoring
box(4.3, 6.55, 5.8, 1.9, CM, BM)
t(7.2, 8.15, '① Gene Signature Scoring',             fontsize=10, fontweight='bold', color=BM)
t(7.2, 7.7,  '7 TLS signatures scored per spot  (Scanpy score_genes)', fontsize=8.5)
t(7.2, 7.3,  'B cell core  ·  CXCL13 anchor  ·  Plasma output',        fontsize=8, color='#555')
t(7.2, 6.9,  'T cell zone  ·  TFH  ·  Chemokines  ·  Tregs',           fontsize=8, color='#555')

# Step 2 — spatial features
box(4.3, 4.3, 5.8, 2.0, CM, BM)
t(7.2, 6.0, '② Spatial Feature Computation',              fontsize=10, fontweight='bold', color=BM)
t(7.2, 5.55, "Moran's I  +  local gradient  per signature", fontsize=8.5)
t(7.2, 5.15, '14 spatial features per spot',               fontsize=9,  fontweight='bold')
t(7.2, 4.7,  'Captures local clustering & boundary sharpness', fontsize=8, color='#555')
t(7.2, 4.35, 'Per-dataset z-score normalization',          fontsize=8,  color='#555')

# Step 3 — clustering
box(4.3, 1.6, 5.8, 2.4, CM, BM)
t(7.2, 3.7,  '③ Unsupervised K-Means Clustering',    fontsize=10, fontweight='bold', color=BM)
t(7.2, 3.25, 'k = 2  ·  n_init = 10  ·  No labels used', fontsize=8.5)
t(7.2, 2.85, 'Combined: 92,684 spots  (RCC + KC/LC)', fontsize=8.5, fontweight='bold')
t(7.2, 2.4,  '→  TLS-associated cluster',            fontsize=9,  fontweight='bold', color=BM)
t(7.2, 2.0,  '→  Non-TLS cluster',                   fontsize=9,  fontweight='bold', color='#555')

# ── arrows: data → methods ────────────────────────────────────────────────
# RCC → step 1 (short horizontal)
arr(3.9, 6.65, 4.3, 7.5)
# KC/LC → step 1 (curved up)
arr(3.9, 2.8, 4.3, 6.55, rad=-0.32)

# between steps
arr(7.2, 6.55, 7.2, 6.3)
arr(7.2, 4.30, 7.2, 4.0)

# ── RESULTS ───────────────────────────────────────────────────────────────
# Result 1 — TLS recovery
box(10.5, 5.7, 6.2, 2.8, CR, BR)
t(13.6, 8.2,  'TLS Maturation State Recovery',           fontsize=10, fontweight='bold', color=BR)
t(13.6, 7.75, 'ARI = 0.437  (KC/LC pathologist labels)',  fontsize=9,  fontweight='bold')
t(13.6, 7.3,  '90.7% of annotated TLS → TLS cluster',    fontsize=8.5)
t(13.6, 6.9,  'ETLS 41.4%   ·   MTLS 49.3%',            fontsize=8,  color='#555')
t(13.6, 6.45, '94.1% of TUM/NOR → non-TLS cluster',     fontsize=8.5)
t(13.6, 6.05, 'KC/LC labels never seen during clustering', fontsize=7.5, color='#999', style='italic')

# Result 2 — benchmark
box(10.5, 2.9, 6.2, 2.55, CR, BR)
t(13.6, 5.15, 'Cross-cancer Benchmark  (KC/LC)',         fontsize=10, fontweight='bold', color=BR)
t(13.6, 4.7,  'K-Means centroid distance   AUROC = 0.859', fontsize=8.5, fontweight='bold')
t(13.6, 4.3,  'Logistic regression                        AUROC = 0.811', fontsize=8.5)
t(13.6, 3.85, 'Permutation null 95th pct              AUROC = 0.518', fontsize=8, color='#888')
t(13.6, 3.35, 'Both methods  p < 0.001',               fontsize=8.5, fontweight='bold', color=BR)

# Result 3 — CXCL13
box(10.5, 0.5, 6.2, 2.15, CR, BR)
t(13.6, 2.35, 'CXCL13 Spatial Decomposition',         fontsize=10, fontweight='bold', color=BR)
t(13.6, 1.9,  '85% of CXCL13⁺ spots outside TLS tissue', fontsize=8.5)
t(13.6, 1.5,  'Exhaustion (HAVCR2/LAG3)  ρ = 0.233',  fontsize=8,  color='#555')
t(13.6, 1.1,  'TFH (CXCR5)                    ρ = 0.039', fontsize=8,  color='#888')
t(13.6, 0.72, 'Independent finding — no model required', fontsize=7.5, color='#999', style='italic')

# ── arrows: methods → results ─────────────────────────────────────────────
arr(10.1, 2.8, 10.5, 7.1,  rad=0.28)  # → result 1
arr(10.1, 2.8, 10.5, 4.17, rad=0.0)   # → result 2
arr(10.1, 2.8, 10.5, 1.6,  rad=-0.18, color='#aaa', lw=1.2)  # → CXCL13 (independent)
ax.text(10.0, 2.05, 'independent\nfinding', ha='right', fontsize=7,
        color='#aaa', style='italic', linespacing=1.3)

plt.tight_layout(pad=0.2)
plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved → {OUT}')
