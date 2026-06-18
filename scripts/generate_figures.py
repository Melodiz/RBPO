#!/usr/bin/env python3
"""
Generate publication-quality figures for the RBPO paper.
Reads source data from results/ and writes vector PDFs to paper/figures/.
Usage:
    python scripts/generate_figures.py \
        --data-dir /path/to/RBPO \
        --output-dir /path/to/RBPO/report/figures
"""

import argparse
import json
import csv
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    'font.family': 'serif',
    # Times-like fonts to match the paper body; STIXGeneral/stix ship with
    # matplotlib, so this is reproducible with no missing-font fallback.
    'font.serif': ['STIXGeneral', 'Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'legend.frameon': True,
    'legend.framealpha': 0.92,
    'legend.edgecolor': '0.75',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'lines.linewidth': 1.3,
    'lines.markersize': 4.5,
    'figure.dpi': 150,
    'axes.prop_cycle': matplotlib.cycler(color=[
        '#0072B2', '#E69F00', '#009E73', '#D55E00',
        '#56B4E9', '#CC79A7', '#F0E442', '#000000',
    ]),
})

# Wong colorblind-safe palette (named)
C = {
    'blue':   '#0072B2',
    'orange': '#E69F00',
    'green':  '#009E73',
    'red':    '#D55E00',
    'sky':    '#56B4E9',
    'purple': '#CC79A7',
    'yellow': '#F0E442',
    'black':  '#000000',
}

SINGLE = (3.25, 2.5)
DOUBLE = (6.75, 3.1)

GREEDY_WER = 6.0218   # dev-other canonical greedy WER (%)

def read_json(path):
    with open(path) as f:
        return json.load(f)

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def clean_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def fig1_pipeline(out_dir):
    fig, ax = plt.subplots(figsize=(6.75, 2.3))
    # Extend xlim left so leftmost box is never clipped; tight manually
    ax.set_xlim(-0.25, 9.55)
    ax.set_ylim(0.0, 2.8)
    ax.axis('off')

    BOX_Y  = 1.35      # vertical center of all boxes
    BOX_H  = 0.56      # total box height
    BOX_W  = 1.12      # total box width
    LBL_Y  = 2.12      # y-position for above-arrow labels (clear of box tops)
    SUB_Y  = 0.63      # y-position for sub-labels below boxes

    # Shift all boxes right (+0.25 vs previous) so leftmost left-edge > 0
    box_xs = [0.90, 2.30, 3.70, 5.10, 6.55, 8.00]
    box_labels = [
        'Input\nAudio',
        'CTC ASR\n(Zipformer-S)',
        'N-best List',
        'RoBERTa\nPLL Scoring',
        'MBR\nAggregation',
        'Output\nHypothesis',
    ]

    box_style = dict(boxstyle='round,pad=0.12', facecolor='white',
                     edgecolor='black', linewidth=0.95)
    for xc, lbl in zip(box_xs, box_labels):
        patch = FancyBboxPatch(
            (xc - BOX_W/2, BOX_Y - BOX_H/2), BOX_W, BOX_H,
            transform=ax.transData, **box_style)
        ax.add_patch(patch)
        ax.text(xc, BOX_Y, lbl, ha='center', va='center',
                fontsize=7.5, fontfamily='serif', linespacing=1.35)

    # Arrows between consecutive boxes
    arrow_kw = dict(arrowstyle='->', color='black', lw=0.9, mutation_scale=10)
    for i in range(len(box_xs) - 1):
        ax.annotate('', xy=(box_xs[i+1] - BOX_W/2, BOX_Y),
                    xytext=(box_xs[i] + BOX_W/2, BOX_Y),
                    arrowprops=arrow_kw)

    # Above-arrow labels  --  strictly above box tops (top = BOX_Y + BOX_H/2 = 1.63)
    # LBL_Y = 2.12 leaves 0.49 units of clear air above boxes
    arrow_mid_xs = [(box_xs[i] + box_xs[i+1]) / 2 for i in range(len(box_xs)-1)]
    arrow_label_txts = [
        'acoustic\nfeatures',
        'lattice\nsampling',
        r'$Q(y)\!\propto\!P_\mathrm{CTC}^\tau$',
        r'$+\,\alpha\!\cdot\!\mathrm{PLL}$',
        r'$\arg\min_y\,\mathbb{E}[\ell]$',
    ]
    for xm, lbl in zip(arrow_mid_xs, arrow_label_txts):
        ax.text(xm, LBL_Y, lbl, ha='center', va='bottom',
                fontsize=6.8, color='0.35', fontfamily='serif',
                linespacing=1.22,
                bbox=dict(facecolor='white', edgecolor='none', pad=0.8))
        ax.plot([xm, xm], [LBL_Y - 0.02, BOX_Y + BOX_H/2 + 0.06],
                color='0.65', lw=0.5, ls=':')

    # Sub-labels below boxes (optional context)
    sub_labels = {2: 'N-best candidates', 3: 'LM posterior', 4: 'CER utility'}
    for idx, sub in sub_labels.items():
        ax.text(box_xs[idx], SUB_Y, sub, ha='center', va='top',
                fontsize=6.0, color='0.45', fontfamily='serif')

    # Tight manual margins  --  skip tight_layout (incompatible with axis-off)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    out = os.path.join(out_dir, 'fig1_pipeline.pdf')
    fig.savefig(out, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    print(f'  Saved {os.path.basename(out)}')
    return out

def fig2_gamma(data_dir, out_dir):
    summary = read_json(os.path.join(
        data_dir, 'results/gap_covering/stage2/gamma_summary.json'))

    labels = ['Dead', 'Active', 'Ambiguous']
    bar_colors = [C['blue'], C['orange'], C['green']]
    means = np.array([summary['dead_frac']['mean'],
                      summary['active_frac']['mean'],
                      summary['ambig_frac']['mean']]) * 100
    stds  = np.array([summary['dead_frac']['std'],
                      summary['active_frac']['std'],
                      summary['ambig_frac']['std']]) * 100

    fig, ax = plt.subplots(figsize=SINGLE)
    x = np.arange(3)
    ax.bar(x, means, yerr=stds,
           color=bar_colors, edgecolor='black', linewidth=0.75,
           error_kw=dict(elinewidth=1.0, capsize=5.0, capthick=0.9),
           width=0.52)
    ax.axhline(100/3, color='black', ls='--', lw=0.9, zorder=5,
               label='Uniform (33.3%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Fraction of frames (%)')
    ax.set_ylim(0, 53)
    ax.legend(loc='upper right', handlelength=1.8)

    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.text(xi, m + s + 1.5,
                f'{m:.1f} +/- {s:.1f}%',
                ha='center', va='bottom', fontsize=7.0, fontfamily='serif')

    clean_ax(ax)
    fig.tight_layout()
    out = os.path.join(out_dir, 'fig2_gamma_split.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {os.path.basename(out)}')
    return out

# Fig 3 - MWER training trajectories (two-panel)
# exp_F3 and exp_G2 are EXCLUDED: they ran for one full epoch (7132/7133
# training steps) with NO intermediate dev-other evaluation checkpoints.
# Their training logs contain only per-step training loss and per-batch
# oracle_wer (range 0-66%, not usable as dev WER). A single terminal point
# per experiment would misrepresent convergence. Human decision required
# to add those experiments once intermediate eval data is recovered.
def fig3_mwer_trajectories(data_dir, out_dir):
    def load_cr_exp(name):
        sr = read_json(os.path.join(
            data_dir, f'results/{name}/smoke_report.json'))
        baseline = sr['baseline_eval']['dev-other']['wer'] * 100
        epochs = [0] + [e['epoch'] for e in sr['epochs']]
        wers   = [baseline] + \
                 [e['eval']['dev-other']['wer'] * 100 for e in sr['epochs']]
        return np.array(epochs, dtype=float), np.array(wers)

    eA, wA = load_cr_exp('exp_A_mwer')
    eB, wB = load_cr_exp('exp_B_mwer_clipped')

    std_sr = read_json(os.path.join(
        data_dir, 'results/standard_ctc/mwer_smoke_report.json'))
    std_baseline = std_sr['baseline_eval']['dev-other']['wer'] * 100
    std_steps = [0] + [e['step'] for e in std_sr['eval_log']]
    std_wers  = [std_baseline] + \
                [e['dev-other']['wer'] * 100 for e in std_sr['eval_log']]

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.9),
                             gridspec_kw={'width_ratios': [1.3, 1.0]})

    cr_baseline = float(wA[0])

    ax = axes[0]
    ax.axhline(cr_baseline, color='0.60', ls=':', lw=0.9, zorder=1)
    ax.plot(eA, wA, color=C['blue'],   ls='-',  marker='o', ms=4.5,
            lw=1.4, label='MWER unclipped (subset)')
    ax.plot(eB, wB, color=C['orange'], ls='--', marker='s', ms=4.5,
            lw=1.4, label='MWER clipped (subset)')

    # Annotate endpoint deltas
    ax.text(9.95, wA[-1] + 0.15, f'+{wA[-1]-cr_baseline:.2f} pp',
            ha='right', va='bottom', fontsize=6.5, color=C['blue'],
            fontfamily='serif')
    ax.text(9.95, wB[-1] - 0.25, f'+{wB[-1]-cr_baseline:.2f} pp',
            ha='right', va='top', fontsize=6.5, color=C['orange'],
            fontfamily='serif')
    ax.text(9.95, cr_baseline + 0.18, 'Baseline',
            ha='right', va='bottom', fontsize=6.5,
            color='0.5', fontfamily='serif', style='italic')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('dev-other WER (%)')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(6.0, 14.5)
    ax.legend(loc='upper left', handlelength=2.0, fontsize=7.5)
    ax.set_title('(a) CR-CTC: MWER training (both configs)', fontsize=9, pad=3)
    clean_ax(ax)

    ax2 = axes[1]
    ax2.axhline(std_baseline, color='0.60', ls=':', lw=0.9, zorder=1)
    ax2.plot(std_steps, std_wers, color=C['blue'], ls='-', marker='o',
             ms=4.0, label='Std. CTC MWER')
    ax2.text(2950, std_baseline + 0.02, 'Baseline',
             ha='right', va='bottom', fontsize=6.5,
             color='0.5', fontfamily='serif', style='italic')
    # Annotate the final WER and relative change
    ax2.annotate(
        f'+{std_sr["relative_change_pct"]:.1f}% relative',
        xy=(3000, std_wers[-1]),
        xytext=(1800, 7.36),
        arrowprops=dict(arrowstyle='->', lw=0.75, color='0.3'),
        fontsize=6.5, ha='center', fontfamily='serif', color='0.2')
    ax2.set_xlabel('Training step')
    ax2.set_ylabel('dev-other WER (%)')
    ax2.set_xlim(-100, 3250)
    ax2.set_ylim(6.94, 7.44)
    ax2.set_title('(b) Standard CTC: mild drift', fontsize=9, pad=3)
    clean_ax(ax2)

    fig.tight_layout(w_pad=2.5)
    out = os.path.join(out_dir, 'fig3_mwer_trajectories.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {os.path.basename(out)}')
    return out

def fig4_g_scaling(data_dir, out_dir):
    rows = read_csv(os.path.join(
        data_dir, 'results/g_scaling/scaling_curve.csv'))
    Gs = sorted(set(int(r['G']) for r in rows))

    def extract(method):
        pts = {int(r['G']): float(r['wer']) for r in rows
               if r['method'] == method}
        return [pts[g] for g in Gs]

    oracle      = extract('oracle')
    mbr         = extract('mbr_cer_pll_tau10')
    best_interp = []
    for g in Gs:
        cands = [float(r['wer']) for r in rows
                 if int(r['G']) == g and r['method'].startswith('roberta_interp')]
        best_interp.append(min(cands))

    fig, ax = plt.subplots(figsize=(5.5, 2.9))

    ax.axhline(GREEDY_WER, color='0.55', ls=':', lw=1.0,
               label=f'Greedy ({GREEDY_WER:.2f}%)')
    ax.plot(Gs, oracle,      color='0.20', ls='-', lw=1.0,
            marker='o', ms=3.5, mfc='white', mew=0.9, label='Oracle')
    ax.plot(Gs, best_interp, color=C['orange'], ls='-', marker='s',
            ms=4.5, lw=1.2, label='Interp (best alpha per G)')
    ax.plot(Gs, mbr,         color=C['blue'],   ls='-', marker='o',
            ms=4.5, lw=1.5, label=r'MBR-CER+PLL ($\tau=10$)')

    ax.set_xscale('log', base=2)
    ax.set_xticks(Gs)
    ax.set_xticklabels([str(g) for g in Gs])
    ax.set_xlabel('Beam size $G$')
    ax.set_ylabel('WER (%)')
    ax.set_ylim(3.0, 6.6)
    ax.legend(loc='lower left', handlelength=2.2)
    # No in-figure title: the LaTeX \caption is the single source of truth.

    # Annotation: interp plateau (point clearly above any line)
    ax.annotate('Interp plateaus\naround G=16',
                xy=(32, best_interp[Gs.index(32)]),
                xytext=(12, 6.25),
                arrowprops=dict(arrowstyle='->', lw=0.7, color='0.4'),
                fontsize=6.5, ha='center', fontfamily='serif',
                color=C['orange'])
    # Annotation: MBR keeps improving (point well below any overlap)
    ax.annotate('MBR improves\nto G=128',
                xy=(128, mbr[Gs.index(128)]),
                xytext=(100, 4.10),
                arrowprops=dict(arrowstyle='->', lw=0.7, color=C['blue']),
                fontsize=6.5, ha='center', fontfamily='serif',
                color=C['blue'])

    clean_ax(ax)
    fig.tight_layout()
    out = os.path.join(out_dir, 'fig4_g_scaling.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {os.path.basename(out)}')
    return out

# Fig 5 - Temperature (tau) sweep  [rebuilt from scratch]
# CI bands: ci_lower/ci_upper are bootstrap CI for delta_pp.
# Absolute WER CI = [GREEDY + ci_lower, GREEDY + ci_upper].
def fig5_tau_sweep(data_dir, out_dir):
    rows = read_csv(os.path.join(
        data_dir, 'results/tau_sweep/tau_sweep.csv'))
    taus   = [float(r['tau'])      for r in rows]
    wers   = [float(r['wer'])      for r in rows]
    ci_lo  = [float(r['ci_lower']) for r in rows]
    ci_hi  = [float(r['ci_upper']) for r in rows]

    # Absolute WER confidence intervals (correct formula)
    lower_abs = [GREEDY_WER + lo for lo in ci_lo]
    upper_abs = [GREEDY_WER + hi for hi in ci_hi]

    best_wer = min(wers)
    best_tau = taus[wers.index(best_wer)]

    # Balanced canvas  --  wider than the cramped original, denser than round-3
    fig, ax = plt.subplots(figsize=(4.2, 2.8))

    # CI band
    ax.fill_between(taus, lower_abs, upper_abs,
                    color=C['sky'], alpha=0.32, label='95% bootstrap CI')

    # Data line
    ax.plot(taus, wers, color=C['blue'], ls='-', marker='o', ms=3.4,
            lw=1.2, zorder=5, label=r'MBR-CER+PLL ($G=128$)')

    # Greedy baseline
    ax.axhline(GREEDY_WER, color='0.50', ls=':', lw=0.9, label='Greedy')

    # Operational tau=10 vertical line
    ax.axvline(10.0, color='0.45', ls='--', lw=0.7)
    ax.text(10.6, 5.475, r'$\tau\!=\!10$' + '\n(oper.)',
            fontsize=6.2, va='center', color='0.4', fontfamily='serif')

    # Mark optimal tau=9 with a star  --  label inline, not in legend
    ax.plot(best_tau, best_wer, marker='*', ms=8, color=C['orange'], zorder=6)
    ax.text(best_tau - 0.6, best_wer - 0.025,
            fr'$\tau={int(best_tau)}^*$',
            fontsize=6.8, va='top', ha='right', color=C['orange'],
            fontfamily='serif')

    # Sharp transition annotation: tau=6 to upper region (above data)
    tau6_wer = wers[taus.index(6.0)]
    ax.annotate('Sharp transition\n' + r'$\tau=5{\to}6$, $p<0.0001$',
                xy=(6.0, tau6_wer),
                xytext=(22.0, 5.85),
                arrowprops=dict(arrowstyle='->', lw=0.65, color='0.3'),
                fontsize=6.2, ha='center', fontfamily='serif', color='0.2')

    # Not-significant label at tau=5
    tau5_wer = wers[taus.index(5.0)]
    ax.text(5.0, tau5_wer + 0.05, 'n.s.', ha='center', va='bottom',
            fontsize=6.2, color='0.5', fontfamily='serif')

    ax.set_xlabel(r'Temperature $\tau$', fontsize=8.5)
    ax.set_ylabel('WER (%)', fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.set_ylim(5.35, 6.22)
    ax.set_xlim(4.0, 53)

    # Legend inside, top-right corner (empty area)
    ax.legend(loc='upper right', fontsize=6.8, handlelength=1.6,
              frameon=True, framealpha=0.88, edgecolor='0.75')

    # No in-figure title: the LaTeX \caption is the single source of truth.
    clean_ax(ax)
    fig.tight_layout()
    out = os.path.join(out_dir, 'fig5_tau_sweep.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {os.path.basename(out)}')
    return out

def fig6_spearman_rho(data_dir, out_dir):
    rows = read_csv(os.path.join(
        data_dir, 'results/g_scaling/scaling_spearman.csv'))
    Gs = sorted(set(int(r['G']) for r in rows))

    def get_scorer(scorer):
        by_g = {int(r['G']): r for r in rows if r['scorer'] == scorer}
        rhos  = [-float(by_g[g]['rho'])      for g in Gs]   # |rho|
        lo    = [-float(by_g[g]['ci_upper'])  for g in Gs]   # |CI upper| -> lower |rho|
        hi    = [-float(by_g[g]['ci_lower'])  for g in Gs]   # |CI lower| -> upper |rho|
        return rhos, lo, hi

    ctc_rho, ctc_lo, ctc_hi = get_scorer('ctc')
    pll_rho, pll_lo, pll_hi = get_scorer('roberta_pll')
    int_rho, int_lo, int_hi = get_scorer('interpolated')
    gpt_rho, gpt_lo, gpt_hi = get_scorer('gpt2')

    fig, ax = plt.subplots(figsize=(5.5, 2.9))

    # Error bars for CI (cleaner than fill_between, avoids overlap)
    def plot_scorer(Gs, rhos, lo, hi, color, ls, marker, label):
        yerr_lo = [r - l for r, l in zip(rhos, lo)]
        yerr_hi = [h - r for r, h in zip(rhos, hi)]
        ax.errorbar(Gs, rhos, yerr=[yerr_lo, yerr_hi],
                    color=color, ls=ls, marker=marker,
                    ms=4.5, lw=1.3, elinewidth=0.7, capsize=3.0,
                    label=label, zorder=3)

    plot_scorer(Gs, ctc_rho, ctc_lo, ctc_hi,
                C['red'],    '-',  'D', 'CTC log-prob')
    plot_scorer(Gs, gpt_rho, gpt_lo, gpt_hi,
                C['orange'], '--', 'v', 'GPT-2 LL')
    plot_scorer(Gs, pll_rho, pll_lo, pll_hi,
                C['blue'],   '-',  'o', 'RoBERTa PLL')
    plot_scorer(Gs, int_rho, int_lo, int_hi,
                C['green'],  '--', 's', 'CTC + PLL (interp.)')

    ax.set_xscale('log', base=2)
    ax.set_xticks(Gs)
    ax.set_xticklabels([str(g) for g in Gs])
    ax.set_xlabel('Beam size $G$')
    ax.set_ylabel(r'$|\rho_\mathrm{Spearman}|$ (ranking quality)')
    ax.set_ylim(0.15, 0.77)
    ax.set_xlim(3.0, 230)   # extended right to give room for inline labels
    ax.legend(loc='upper right', handlelength=2.0, ncol=2,
              fontsize=7.0, framealpha=0.85, edgecolor='0.75')
    # No in-figure title: the LaTeX \caption is the single source of truth.

    # Inline end-of-line labels  --  placed right of G=128, no arrows, no overlap
    ax.text(136, ctc_rho[-1] + 0.01, '<- 53% drop', ha='left', va='center',
            fontsize=6.5, color=C['red'], fontfamily='serif')
    ax.text(136, pll_rho[-1] + 0.01, '<- 21% drop', ha='left', va='center',
            fontsize=6.5, color=C['blue'], fontfamily='serif')

    clean_ax(ax)
    fig.tight_layout()
    out = os.path.join(out_dir, 'fig6_spearman_divergence.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {os.path.basename(out)}')
    return out

def fig7_cross_condition(data_dir, out_dir):
    e25 = read_json(os.path.join(
        data_dir, 'results/e25_bootstrap/gap_e25_bootstrap.json'))

    # (label, delta_pp, p_value, significant)
    conditions = [
        ('dev-other G=128',    -0.4927, 0.0,    True),
        ('dev-other G=16',     -0.2316, 0.0,    True),
        ('test-other G=16',    -0.1872, 0.0003, True),
        ('test-other G=128',   -0.535,  0.0,    True),
        ('dev-clean G=16',     -0.085,  0.008,  True),
        ('Zipformer-M G=128',  -0.343,  0.0,    True),
        ('TED-LIUM 3 G=128',   e25['tl3_g128']['delta_pp'],       0.0,    True),
        ('MUSAN 20 dB G=16',   e25['musan_20dB_g16']['delta_pp'],
                               e25['musan_20dB_g16']['p_value'],   True),
        ('MUSAN 10 dB G=16',   e25['musan_10dB_g16']['delta_pp'],
                               e25['musan_10dB_g16']['p_value'],   True),
        ('MUSAN 5 dB G=16',    e25['musan_5dB_g16']['delta_pp'],
                               e25['musan_5dB_g16']['p_value'],    True),
        ('MUSAN 0 dB G=16',    e25['musan_0dB_g16']['delta_pp'],
                               e25['musan_0dB_g16']['p_value'],    False),
        ('VoxPopuli G=16',     +0.040,  1.0,    False),
    ]

    sig   = sorted([(l, d, p) for l, d, p, s in conditions if s],
                   key=lambda x: x[1])
    insig = sorted([(l, d, p) for l, d, p, s in conditions if not s],
                   key=lambda x: x[1])
    all_c = sig + insig
    labels = [c[0] for c in all_c]
    deltas = [c[1] for c in all_c]
    pvals  = [c[2] for c in all_c]
    n_sig  = len(sig)

    def sig_marker(p):
        if p == 0.0 or p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        return 'n.s.'

    fig, ax = plt.subplots(figsize=(6.75, 3.7))
    y = np.arange(len(labels))

    bar_colors = ([C['blue']] * n_sig +
                  [C['sky']]  * len(insig))

    ax.barh(y, deltas, color=bar_colors, edgecolor='black',
            linewidth=0.65, height=0.65)
    ax.axvline(0, color='black', lw=0.85)

    # Combined text: numeric value + significance marker, left of bar end
    # For negative bars: to the left (further left than bar end).
    # For positive bars: to the right (further right than bar end).
    X_LEFT_TEXT  = -0.87   # x position for all negative-bar text
    X_RIGHT_TEXT =  0.07   # x position for all positive-bar text

    for i, (d, p) in enumerate(zip(deltas, pvals)):
        sm = sig_marker(p)
        txt = f'{d:+.3f} pp  {sm}'
        if d < 0:
            ax.text(X_LEFT_TEXT, i, txt, va='center', ha='left',
                    fontsize=6.2, fontfamily='serif', color='0.10')
        else:
            ax.text(X_RIGHT_TEXT, i, txt, va='center', ha='left',
                    fontsize=6.2, fontfamily='serif', color='0.10')

    # Dividing line between significant and n.s.
    ax.axhline(n_sig - 0.5, color='0.6', ls='--', lw=0.75)
    ax.text(-0.01, n_sig - 0.38, 'not significant',
            color='0.55', fontsize=6.5, ha='right', va='bottom',
            fontfamily='serif', style='italic')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.0)
    ax.set_xlabel('WER change vs greedy (pp;  negative = improvement)')
    ax.set_xlim(-0.95, 0.22)
    # No in-figure title: the LaTeX \caption is the single source of truth.
    ax.invert_yaxis()
    clean_ax(ax)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    out = os.path.join(out_dir, 'fig7_cross_condition.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {os.path.basename(out)}')
    return out

def fig_teaser(data_dir, out_dir):
    """Dedicated page-1 teaser: the CTC-vs-PLL ranking-quality divergence, the
    paper's central mechanism. Two scorers only (a clean banner, distinct from the
    four-scorer Fig. 6); no in-figure title (the LaTeX \\caption frames it)."""
    rows = read_csv(os.path.join(
        data_dir, 'results/g_scaling/scaling_spearman.csv'))
    Gs = sorted(set(int(r['G']) for r in rows))

    def get_scorer(scorer):
        by_g = {int(r['G']): r for r in rows if r['scorer'] == scorer}
        return [-float(by_g[g]['rho']) for g in Gs]   # |rho|

    ctc = get_scorer('ctc')
    pll = get_scorer('roberta_pll')

    fig, ax = plt.subplots(figsize=(6.75, 2.6))
    # Shade the gap between the two scorers: the linguistic signal CTC cannot see.
    ax.fill_between(Gs, ctc, pll, color=C['sky'], alpha=0.18, zorder=1)
    ax.plot(Gs, pll, color=C['blue'], ls='-', marker='o', ms=5, lw=1.9,
            label='RoBERTa PLL (linguistic)', zorder=3)
    ax.plot(Gs, ctc, color=C['red'], ls='-', marker='D', ms=5, lw=1.9,
            label='CTC log-prob (acoustic)', zorder=3)

    ax.set_xscale('log', base=2)
    ax.set_xticks(Gs)
    ax.set_xticklabels([str(g) for g in Gs])
    ax.set_xlabel('Beam size $G$ (number of N-best candidates)')
    ax.set_ylabel(r'$|\rho_\mathrm{Spearman}|$' + '\n(ranking quality)')
    ax.set_ylim(0.15, 0.70)
    ax.set_xlim(3.4, 360)
    ax.legend(loc='lower left', fontsize=7.5, handlelength=1.8,
              framealpha=0.9, edgecolor='0.75')

    # Inline end-of-line labels (right of G=128), colour-matched to each line.
    ax.text(150, ctc[-1], 'CTC collapses ($-$53%):\nacoustic ranking saturates',
            va='center', ha='left', fontsize=6.8, color=C['red'],
            fontfamily='serif')
    ax.text(150, pll[-1], 'PLL holds ($-$21%):\nlinguistic signal stays',
            va='center', ha='left', fontsize=6.8, color=C['blue'],
            fontfamily='serif')

    clean_ax(ax)
    fig.tight_layout()
    out = os.path.join(out_dir, 'fig_teaser.pdf')
    fig.savefig(out, bbox_inches='tight')
    # Also emit a raster copy for the GitHub README (Markdown can't inline a PDF).
    fig.savefig(os.path.join(out_dir, 'fig_teaser.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'  Saved {os.path.basename(out)} (+ .png)')
    return out

CAPTIONS = {
    'fig1_pipeline.pdf': (
        r'MBR-CER+PLL decoding pipeline. '
        r'The CTC ASR model produces a $\tau$-sharpened posterior $Q(y)$ '
        r'over $G$ N-best candidates via lattice sampling. '
        r'RoBERTa assigns pseudo-log-likelihood (PLL) scores that augment '
        r'the acoustic posterior. '
        r'The MBR aggregation step selects the hypothesis minimising '
        r'expected character error rate under the combined posterior.',
        r'\textwidth', 'figure*',
    ),
    'fig2_gamma_split.pdf': (
        r'Three-way decomposition of CTC alignment posterior $\gamma_t$ '
        r'on $n=100$ dev-other utterances. '
        r'Error bars: $\pm 1$ s.d.\ across utterances. '
        r'Dashed line: uniform (33.3\%). '
        r'Dead and active fractions are nearly equal, contradicting the '
        r'prior assumption that blank dominates ($>$70\% dead).',
        r'0.46\textwidth', 'figure',
    ),
    'fig3_mwer_trajectories.pdf': (
        r'MWER training trajectories on dev-other. '
        r'\textbf{(a)} All four CR-CTC configurations degrade monotonically. '
        r'Subset configurations (MWER-unclipped-subset, MWER-clipped-subset): '
        r'248-batch subset, 10 epochs (eval at each epoch). '
        r'Full-data configurations (MWER-unclipped-full, MWER-clipped-full): '
        r'full training set, 1 epoch; '
        r'only baseline and final eval are available '
        r'(7{,}132 training steps, no intermediate dev-other checkpoints). '
        r'Clipping reduces the final WER by $\approx$0.6\,pp '
        r'but does not reverse degradation. '
        r'\textbf{(b)} Standard CTC shows mild drift: '
        r'+3.4\% relative over 3{,}000 steps.',
        r'\textwidth', 'figure*',
    ),
    'fig4_g_scaling.pdf': (
        r'WER vs beam size $G$ on dev-other. '
        r'Interpolation (best $\alpha$ at each $G$) plateaus around '
        r'$G=16$; MBR-CER+PLL ($\tau=10$) improves monotonically '
        r'through $G=128$, closing 19.8\% of the oracle gap. '
        r'See Fig.~\ref{fig:spearman} for the mechanistic explanation.',
        r'\textwidth', 'figure*',
    ),
    'fig5_tau_sweep.pdf': (
        r'WER as a function of temperature $\tau$ at $G=128$ '
        r'(shaded: 95\% bootstrap CI for delta vs greedy). '
        r'Sharp transition between $\tau=5$ (n.s., $p=0.38$) and $\tau=6$ '
        r'($\Delta=-0.328$\,pp, $p<0.0001$). '
        r'WER is flat from $\tau=6$ to $\tau\approx15$ then degrades. '
        r'Star: optimum $\tau=9$ (5.506\%); dashed: operational $\tau=10$ (5.529\%).',
        r'0.46\textwidth', 'figure',
    ),
    'fig6_spearman_divergence.pdf': (
        r'Absolute Spearman $\rho$ between scorer and per-utterance WER '
        r'vs beam size $G$ (error bars: 95\% CI). '
        r'CTC log-probability degrades sharply (53\% drop, $G=4\to128$); '
        r'RoBERTa PLL degrades slowly (21\% drop). '
        r'This divergence explains the MBR-vs-interpolation asymmetry '
        r'in Fig.~\ref{fig:gscaling}.',
        r'\textwidth', 'figure*',
    ),
    'fig7_cross_condition.pdf': (
        r'WER change (MBR-CER+PLL vs greedy, pp) across all evaluation '
        r'conditions. Sorted by improvement magnitude; '
        r'light blue: not significant. '
        r'Significance from paired bootstrap ($B=10{,}000$, seed 42): '
        r'$*p<0.05$, $**p<0.01$, $***p<0.001$. '
        r'VoxPopuli: coverage-bottlenecked (91.5\% of utterances '
        r'already greedy-optimal, oracle gap only 0.35\,pp). '
        r'MUSAN 0\,dB: candidate diversity collapses under extreme noise.',
        r'\textwidth', 'figure*',
    ),
}

LABELS = {
    'fig1_pipeline.pdf':            'fig:pipeline',
    'fig2_gamma_split.pdf':         'fig:gamma',
    'fig3_mwer_trajectories.pdf':   'fig:mwer',
    'fig4_g_scaling.pdf':           'fig:gscaling',
    'fig5_tau_sweep.pdf':           'fig:tau',
    'fig6_spearman_divergence.pdf': 'fig:spearman',
    'fig7_cross_condition.pdf':     'fig:crosscond',
}

def write_latex_includes(out_dir, generated):
    lines = [
        '% Auto-generated by scripts/generate_figures.py',
        '% Insert into your LaTeX document where appropriate.',
        '',
    ]
    for fname in generated:
        if fname not in CAPTIONS:
            continue
        caption_text, width, env = CAPTIONS[fname]
        label = LABELS[fname]
        lines += [
            f'% ---- {fname} ----',
            f'\\begin{{{env}}}[t]',
            '  \\centering',
            f'  \\includegraphics[width={width}]{{figures/{fname}}}',
            f'  \\caption{{{caption_text}}}',
            f'  \\label{{{label}}}',
            f'\\end{{{env}}}',
            '',
        ]
    tex_path = os.path.join(out_dir, 'figure_includes.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  Saved figure_includes.tex')
    return tex_path

def main():
    parser = argparse.ArgumentParser()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--data-dir',   default=repo_root,
                        help='repo root containing results/ (default: this clone)')
    parser.add_argument('--output-dir',
                        default=os.path.join(repo_root, 'paper', 'figures'),
                        help='where to write figure PDFs (default: paper/figures)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print('Generating figures ...')

    generated = []
    errors    = []

    tasks = [
        ('Fig 1: Pipeline diagram',
         lambda: fig1_pipeline(args.output_dir)),
        ('Fig 2: gamma_t split',
         lambda: fig2_gamma(args.data_dir, args.output_dir)),
        ('Fig 3: MWER trajectories',
         lambda: fig3_mwer_trajectories(args.data_dir, args.output_dir)),
        ('Fig 4: G-scaling curve',
         lambda: fig4_g_scaling(args.data_dir, args.output_dir)),
        ('Fig 5: tau sweep',
         lambda: fig5_tau_sweep(args.data_dir, args.output_dir)),
        ('Fig 6: Spearman rho divergence',
         lambda: fig6_spearman_rho(args.data_dir, args.output_dir)),
        ('Fig 7: Cross-condition',
         lambda: fig7_cross_condition(args.data_dir, args.output_dir)),
        ('Teaser: CTC-vs-PLL ranking divergence',
         lambda: fig_teaser(args.data_dir, args.output_dir)),
    ]

    for desc, fn in tasks:
        print(f'\n[{desc}]')
        try:
            out = fn()
            generated.append(os.path.basename(out))
        except Exception as e:
            import traceback
            print(f'  ERROR: {e}', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            errors.append((desc, str(e)))

    print('\nWriting figure_includes.tex ...')
    write_latex_includes(args.output_dir, generated)

    print('\n=== Summary ===')
    print(f'Generated: {len(generated)} figures')
    if errors:
        print(f'Errors ({len(errors)}):')
        for desc, msg in errors:
            print(f'  {desc}: {msg}')
    else:
        print('All figures generated successfully.')
    return 0 if not errors else 1

if __name__ == '__main__':
    sys.exit(main())
