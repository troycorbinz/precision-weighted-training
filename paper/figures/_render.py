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
    data = json.loads((HERE.parent / 'data' / 'phase3_layer_divergence.json').read_text())
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


# ─── Figure 2: Loss-vs-preference (the paper's central claim) ─────────

def render_fig2():
    """Render the two-panel loss-vs-preference figure for §6.2."""
    data = json.loads((HERE / 'fig2_data.json').read_text())
    pa = data['panel_a_loss_curves']
    pb = data['panel_b_preference']

    bl_traj = pa['baseline']['trajectory']
    gn_traj = pa['gain']['trajectory']
    bl_steps = np.array([p['step'] for p in bl_traj])
    bl_vals = np.array([p['val_loss_smoothed'] for p in bl_traj])
    gn_steps = np.array([p['step'] for p in gn_traj])
    gn_vals = np.array([p['val_loss_smoothed'] for p in gn_traj])

    bl_color = '#777777'
    gn_color = '#7c83ff'

    fig = plt.figure(figsize=(11.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.22,
                          left=0.07, right=0.97, top=0.78, bottom=0.12)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    # ── Left panel: smoothed val-loss curves ──
    ax_l.plot(bl_steps, bl_vals, color=bl_color, lw=2.0,
              label=pa['baseline']['label'])
    ax_l.plot(gn_steps, gn_vals, color=gn_color, lw=2.0,
              label=pa['gain']['label'])
    # Endpoint dots so the reader can see both lines reach step 30K
    ax_l.plot(bl_steps[-1], bl_vals[-1], 'o', color=bl_color, ms=4)
    ax_l.plot(gn_steps[-1], gn_vals[-1], 'o', color=gn_color, ms=4)

    # Endpoint value labels with a Δ bracket
    final_bl = bl_vals[-1]
    final_gn = gn_vals[-1]
    end_x = bl_steps[-1]
    ax_l.annotate(f'{final_gn:.3f}', xy=(end_x, final_gn),
                  xytext=(8, 4), textcoords='offset points',
                  color=gn_color, fontsize=9, fontweight='bold',
                  va='bottom', ha='left')
    ax_l.annotate(f'{final_bl:.3f}', xy=(end_x, final_bl),
                  xytext=(8, -4), textcoords='offset points',
                  color=bl_color, fontsize=9, fontweight='bold',
                  va='top', ha='left')
    # Δ bracket: tiny vertical bracket spanning the two endpoints
    bracket_x = end_x + 1100
    ax_l.plot([bracket_x, bracket_x + 250, bracket_x + 250, bracket_x],
              [final_gn, final_gn, final_bl, final_bl],
              color='#333', lw=0.8, clip_on=False)
    ax_l.text(bracket_x + 600, (final_gn + final_bl) / 2,
              r'$\Delta = 0.004$',
              fontsize=9, color='#333', va='center', ha='left',
              clip_on=False)

    ax_l.set_xlabel('Training step')
    ax_l.set_ylabel('Smoothed val loss')
    ax_l.set_title('Validation loss is the same\n'
                   f'Baseline {final_bl:.3f} · Gain {final_gn:.3f} · '
                   f'difference {abs(final_gn - final_bl):.3f} nats',
                   loc='left', pad=10)
    ax_l.set_xlim(0, 31000)
    ax_l.set_ylim(pa['y_range_suggested'][0], pa['y_range_suggested'][1])
    ax_l.xaxis.set_major_locator(MultipleLocator(5000))
    ax_l.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: '500' if x == 500 else f'{int(x / 1000):,}K' if x >= 1000 else f'{int(x)}'))
    ax_l.set_xticks([500, 5000, 10000, 15000, 20000, 25000, 30000])
    ax_l.grid(axis='y', alpha=0.22, linestyle='--', linewidth=0.5)
    ax_l.legend(loc='upper right', handlelength=2.2)

    # ── Right panel: stacked decisive bar + sensitivity whisker ──
    bl_pct = pb['breakdown_of_decisive']['baseline_pct']
    gn_pct = pb['breakdown_of_decisive']['gain_pct']
    bl_bar_color = pb['breakdown_of_decisive']['baseline_color_hex']
    gn_bar_color = pb['breakdown_of_decisive']['gain_color_hex']
    band = pb['sensitivity_band']
    band_min = band['min_pct']
    band_max = band['max_pct']
    headline = band['headline_pct']

    bar_y = 0.78
    bar_h = 0.18
    whisker_y = 0.42

    # Stacked decisive bar (40.1% baseline | 59.9% gain)
    ax_r.barh(bar_y, bl_pct, height=bar_h, left=0,
              color=bl_bar_color, edgecolor='none')
    ax_r.barh(bar_y, gn_pct, height=bar_h, left=bl_pct,
              color=gn_bar_color, edgecolor='none')

    # In-bar labels
    ax_r.text(bl_pct / 2, bar_y, f'{bl_pct:.1f}% baseline',
              va='center', ha='center', color='white', fontsize=10,
              fontweight='bold')
    ax_r.text(bl_pct + gn_pct / 2, bar_y, f'{gn_pct:.1f}% gain',
              va='center', ha='center', color='white', fontsize=10,
              fontweight='bold')

    # Sensitivity-filter whisker
    ax_r.plot([band_min, band_max], [whisker_y, whisker_y],
              color='#333', lw=1.6, solid_capstyle='butt')
    # End caps
    cap_h = 0.04
    for x in (band_min, band_max):
        ax_r.plot([x, x], [whisker_y - cap_h, whisker_y + cap_h],
                  color='#333', lw=1.6)
    # Headline tick (vertical, brand color)
    ax_r.plot([headline, headline],
              [whisker_y - cap_h * 1.3, whisker_y + cap_h * 1.3],
              color=gn_color, lw=2.4)
    # Only label the min and max — the headline (59.9%) is already
    # prominent in the stacked bar above, and the brand-color tick marks
    # its position on the whisker. Three labels in a 4-point-wide band
    # collide; two labels with horizontal anchoring don't.
    label_y = whisker_y + cap_h + 0.030
    ax_r.text(band_min, label_y, f'{band_min:.1f}% ',
              va='bottom', ha='right', fontsize=9, color='#333')
    ax_r.text(band_max, label_y, f' {band_max:.1f}%',
              va='bottom', ha='left', fontsize=9, color='#333')
    ax_r.text((band_min + band_max) / 2, whisker_y - cap_h - 0.030,
              'Range across all sensitivity filters',
              va='top', ha='center', fontsize=8.5, color='#555',
              fontstyle='italic')

    # Chance reference line at 50%
    ax_r.axvline(50, color='#555', lw=0.9, linestyle='--', ymin=0.05, ymax=0.95)
    ax_r.text(50, 0.06, 'Chance', va='center', ha='center', fontsize=8.5,
              color='#555',
              bbox=dict(boxstyle='round,pad=0.25', fc='white',
                        ec='none'))

    ax_r.set_xlim(0, 100)
    ax_r.set_ylim(0, 1)
    ax_r.set_xlabel('Percent of decisive judgments')
    ax_r.set_yticks([])
    ax_r.spines['left'].set_visible(False)
    ax_r.set_title('Blind A/B preference is decisively for gain\n'
                   '1,181 judgments · 42 judges (29 humans + 13 FMs, 11 vendors) · '
                   r'p = 2.80 × 10$^{-8}$',
                   loc='left', pad=10)
    ax_r.xaxis.set_major_locator(MultipleLocator(20))

    # Suptitle banner across both panels (positioned via subplots_adjust above)
    fig.suptitle('Aggregate val loss says nothing changed; 42 blind judges said something changed.',
                 fontsize=13, fontweight='bold', y=0.96, x=0.5)

    out = HERE / 'fig2_loss_vs_preference.png'
    fig.savefig(out)
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    render_fig1()
    render_fig2()
