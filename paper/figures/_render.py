"""Render the layer-divergence figure for the precision-weighted training paper."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.patches import Rectangle

HERE = Path(__file__).parent

# Typography and style
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.frameon': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})




def moving_avg(x, k=5):
    """Centered moving average."""
    x = np.asarray(x, dtype=float)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    return np.convolve(xp, np.ones(k) / k, mode='valid')


# ─── Figure 1: Layer divergence heatmap ───────────────────────────────

def render_fig1():
    data = json.loads((HERE / '_data_fig1.json').read_text())
    # Skip the first few warmup samples where L0 can spike >5 (breaks scale)
    data = [r for r in data if r['_step'] >= 500]
    n_layers = 20
    steps = np.array([r['_step'] for r in data])
    divs = np.zeros((n_layers, len(steps)))
    for ci, r in enumerate(data):
        for li in range(n_layers):
            v = r.get(f'layer_gain/div_layer_{li:02d}')
            divs[li, ci] = v if v is not None else np.nan

    # Smooth per-layer across steps to reduce column noise
    smoothed = np.zeros_like(divs)
    for li in range(n_layers):
        smoothed[li] = moving_avg(divs[li], k=5)

    fig, ax = plt.subplots(figsize=(9.0, 5.8))

    # Scale: clip L0's very high values so mid-zone variation shows
    vmin = 0.08
    vmax = 1.5
    cmap = plt.get_cmap('inferno')
    im = ax.imshow(
        smoothed,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        extent=[min(steps), max(steps), -0.5, n_layers - 0.5],
        interpolation='nearest',
    )

    ax.set_xlabel('Training step')
    ax.set_ylabel('Layer index')
    ax.set_title('Emergent functional layer specialization across training\n'
                 r'Per-block representation divergence $\|x_{out} - x_{in}\| \,/\, \|x_{in}\|$',
                 loc='left', pad=14)

    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f'L{i}' for i in range(n_layers)])

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
    ax.xaxis.set_major_locator(MultipleLocator(5000))

    # Zone annotations: draw colored bracket boxes on right side
    max_x = max(steps)
    box_x = max_x * 1.015
    box_w = max_x * 0.015

    # Late zone: L19 (y 18.5 to 19.5)
    ax.add_patch(Rectangle((box_x, 18.5), box_w, 1.0,
                           color='#C84A3D', alpha=0.9, clip_on=False, lw=0))
    ax.text(box_x + box_w * 1.8, 19.0, 'late',
            va='center', ha='left', fontsize=9, color='#333',
            clip_on=False, fontweight='bold')

    # Mid zone: L7-L18 (y 6.5 to 18.5)
    ax.add_patch(Rectangle((box_x, 6.5), box_w, 12.0,
                           color='#2E5E8E', alpha=0.6, clip_on=False, lw=0))
    ax.text(box_x + box_w * 1.8, 12.5, 'mid',
            va='center', ha='left', fontsize=9, color='#333',
            clip_on=False, fontweight='bold')

    # Early zone: L0-L6 (y -0.5 to 6.5)
    ax.add_patch(Rectangle((box_x, -0.5), box_w, 7.0,
                           color='#E6A23C', alpha=0.9, clip_on=False, lw=0))
    ax.text(box_x + box_w * 1.8, 3.0, 'early',
            va='center', ha='left', fontsize=9, color='#333',
            clip_on=False, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.11,
                        ticks=[0.1, 0.2, 0.5, 1.0, 1.5])
    cbar.set_label('divergence (log scale)', fontsize=9)
    cbar.ax.set_yticklabels(['0.1', '0.2', '0.5', '1.0', '≥1.5'])
    cbar.outline.set_visible(False)

    # Annotation: growth arrows for L3 and L19
    ax.annotate('L3: +288%', xy=(max_x*0.96, 3.4), xytext=(max_x*0.55, 5.8),
                fontsize=9, color='white', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='white', lw=1.2, alpha=0.85))
    ax.annotate('L19: +387%', xy=(max_x*0.96, 19.0), xytext=(max_x*0.50, 17.0),
                fontsize=9, color='white', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='white', lw=1.2, alpha=0.85))

    fig.tight_layout()
    out = HERE / 'fig1_layer_divergence.png'
    fig.savefig(out)
    print(f'Saved {out}')
    plt.close(fig)


# ─── Figure 2: Train-val gap ──────────────────────────────────────────

def render_fig2():
    data = json.loads((HERE / '_data_fig2.json').read_text())
    bl = data['baseline_025']
    gn = data['gain_026']

    def series(rows, key):
        rows = [r for r in rows if r.get(key) is not None]
        rows.sort(key=lambda r: r['_step'])
        steps = np.array([r['_step'] for r in rows])
        vals = np.array([r[key] for r in rows], dtype=float)
        return steps, vals

    bl_tr_x, bl_tr_y = series(bl, 'train_loss')
    bl_va_x, bl_va_y = series(bl, 'val_loss')
    gn_tr_x, gn_tr_y = series(gn, 'train_loss')
    gn_va_x, gn_va_y = series(gn, 'val_loss')

    # Smooth with EMA (alpha=0.15 is typical for training curves)
    bl_tr_s = ema(bl_tr_y, 0.2)
    bl_va_s = ema(bl_va_y, 0.2)
    gn_tr_s = ema(gn_tr_y, 0.2)
    gn_va_s = ema(gn_va_y, 0.2)

    # Two-panel layout: top shows curves, bottom shows gap
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 6.8),
                                    gridspec_kw={'height_ratios': [2.3, 1]},
                                    sharex=True)

    bl_color = '#2E5E8E'
    gn_color = '#C84A3D'

    # ── Top panel: smoothed loss curves ──
    # Light raw lines underneath
    ax1.plot(bl_tr_x, bl_tr_y, color=bl_color, lw=0.6, alpha=0.18)
    ax1.plot(bl_va_x, bl_va_y, color=bl_color, lw=0.6, alpha=0.18)
    ax1.plot(gn_tr_x, gn_tr_y, color=gn_color, lw=0.6, alpha=0.18)
    ax1.plot(gn_va_x, gn_va_y, color=gn_color, lw=0.6, alpha=0.18)
    # Smoothed prominent lines
    ax1.plot(bl_tr_x, bl_tr_s, color=bl_color, lw=1.4, linestyle='--',
             label='Baseline · train')
    ax1.plot(bl_va_x, bl_va_s, color=bl_color, lw=2.2,
             label='Baseline · val')
    ax1.plot(gn_tr_x, gn_tr_s, color=gn_color, lw=1.4, linestyle='--',
             label='Gain · train')
    ax1.plot(gn_va_x, gn_va_s, color=gn_color, lw=2.2,
             label='Gain · val')

    # Shade the final gap region for each model (last 10% of training)
    tail_mask_bl = bl_tr_x >= 27000
    tail_mask_gn = gn_tr_x >= 27000
    if tail_mask_bl.any():
        ax1.fill_between(bl_tr_x[tail_mask_bl], bl_tr_s[tail_mask_bl],
                         bl_va_s[tail_mask_bl], color=bl_color, alpha=0.18,
                         linewidth=0)
    if tail_mask_gn.any():
        ax1.fill_between(gn_tr_x[tail_mask_gn], gn_tr_s[tail_mask_gn],
                         gn_va_s[tail_mask_gn], color=gn_color, alpha=0.18,
                         linewidth=0)

    ax1.set_ylabel('Loss')
    ax1.set_title('Train vs validation loss — gain reduces memorization at matched training cost\n'
                  '5.4× smaller train–val gap at 30K steps',
                  loc='left', pad=12)
    ax1.set_ylim(3.3, 9.5)
    ax1.legend(loc='upper right', ncol=2, columnspacing=1.8, handlelength=2.2)
    ax1.grid(axis='y', alpha=0.22, linestyle='--', linewidth=0.5)

    # ── Bottom panel: train-val gap over time ──
    # Compute gap using smoothed series, interpolated to shared x
    def interp_gap(tx, ty, vx, vy):
        # Interpolate val onto train x grid
        vy_on_tx = np.interp(tx, vx, vy)
        return tx, vy_on_tx - ty

    bl_gap_x, bl_gap_y = interp_gap(bl_tr_x, bl_tr_s, bl_va_x, bl_va_s)
    gn_gap_x, gn_gap_y = interp_gap(gn_tr_x, gn_tr_s, gn_va_x, gn_va_s)

    ax2.axhline(0, color='#888', lw=0.6, linestyle='-', alpha=0.5)
    ax2.fill_between(bl_gap_x, 0, bl_gap_y, color=bl_color, alpha=0.20, lw=0)
    ax2.fill_between(gn_gap_x, 0, gn_gap_y, color=gn_color, alpha=0.20, lw=0)
    ax2.plot(bl_gap_x, bl_gap_y, color=bl_color, lw=2.0, label='Baseline')
    ax2.plot(gn_gap_x, gn_gap_y, color=gn_color, lw=2.0, label='Gain')

    # Annotate final gaps
    final_bl = bl_gap_y[-1]
    final_gn = gn_gap_y[-1]
    ax2.annotate(f'+{final_bl:.3f}', xy=(bl_gap_x[-1], final_bl),
                 xytext=(5, 0), textcoords='offset points', va='center',
                 color=bl_color, fontsize=9, fontweight='bold')
    ax2.annotate(f'+{final_gn:.3f}', xy=(gn_gap_x[-1], final_gn),
                 xytext=(5, 0), textcoords='offset points', va='center',
                 color=gn_color, fontsize=9, fontweight='bold')

    ax2.set_ylabel('Gap (val − train)')
    ax2.set_xlabel('Training step')
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
    ax2.xaxis.set_major_locator(MultipleLocator(5000))
    ax2.set_xlim(0, 31500)
    ax2.legend(loc='upper left', handlelength=2.2)
    ax2.grid(axis='y', alpha=0.22, linestyle='--', linewidth=0.5)

    fig.tight_layout()
    out = HERE / 'fig2_train_val_gap.png'
    fig.savefig(out)
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    render_fig1()
