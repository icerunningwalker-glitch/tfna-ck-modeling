"""Render the transport model's time-depth concentration map."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import figure_style as pf


def draw_depth_and_exposure(m, times_h, states, outputs):
    p = m.p
    concentration = np.maximum(m.contact(states), 0)
    cmap = LinearSegmentedColormap.from_list(
        'skin_concentration', [(0, '#fffaf5'), (.25, pf.C[3]),
                               (.55, pf.C[0]), (.80, pf.C[4]), (1, pf.C[2])], N=256)
    fig = plt.figure(figsize=(12.0, 4.1), layout='constrained')
    grid = fig.add_gridspec(1, 3, width_ratios=[1, .045, 1], wspace=.10)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    right = fig.add_subplot(grid[0, 2])
    # Cell edges preserve the finite-volume depth geometry; no invented SC data.
    time_edges = np.r_[times_h[0], (times_h[:-1] + times_h[1:]) / 2, times_h[-1]]
    mesh = ax.pcolormesh(
        time_edges, m.edges, concentration.T,
        cmap=cmap, vmin=0, vmax=.18,
        shading='flat', rasterized=True)
    bar = fig.colorbar(mesh, cax=cax, ticks=np.arange(0, .181, .03))
    bar.set_label('Free intact tFNA (µmol/L)', fontsize=9)
    bar.ax.tick_params(labelsize=8)
    ax.axhspan(0, p.sc_um, color='#eeeeeb', zorder=3)
    for depth in [p.sc_um, p.ve_bottom_um]:
        ax.axhline(depth, color='white', linestyle='--', linewidth=1.5, zorder=4)
    for depth, label in [(p.ve_bottom_um + 18, 'Dermis'),
                         ((p.sc_um + p.ve_bottom_um) / 2, 'VE')]:
        ax.text(3.90, depth, label, ha='right', va='center', color='white', fontsize=9)
    ax.text(3.90, p.sc_um / 2, 'SC', ha='right', va='center',
            color='#333333', fontsize=7)
    ax.set(xlabel='Time from application (h)', ylabel='Depth (µm)',
           xlim=(0, 4), ylim=(0, p.bottom_um), xticks=[0, 1, 2, 3, 4],
           yticks=[0, 100, 200, 300, 400], title='Free tFNA concentration: time–depth map')
    pf.lab(ax, 'A')
    q = outputs[outputs.time_min <= 240]
    for layer, col, ls, label in [
        ('VE', pf.C[2], '-', 'Viable epidermis'),
        ('dermis', pf.C[1], '--', 'Shallow dermis')]:
        right.plot(q.time_h, q[layer + '_contact_uM'], color=col, ls=ls, lw=2, label=label)
    right.axvspan(0, p.application_min / 60, color=pf.C[3], alpha=.5)
    right.set(xlabel='Time from application (h)',
              ylabel='Cell-contact tFNA concentration (µmol/L)',
              title='Residual delivery after mask removal', xlim=(0, 4),
              xticks=[0, 1, 2, 3, 4], ylim=(0, .18), yticks=np.arange(0, .181, .03))
    right.text(.02, .96, 'Team tFNA limit: 0.5 µmol/L (off scale)',
               transform=right.transAxes, va='top', fontsize=8, color=pf.C[1])
    stop_h = p.application_min / 60
    right.axvline(stop_h, ymin=0, ymax=.90, color='#77745c',
                  linewidth=.85, linestyle=(0, (3, 3)))
    right.annotate(f'Stop dosing\n{p.application_min:g} min',
                   xy=(stop_h, .123), xytext=(stop_h + .27, .144),
                   fontsize=9, color=pf.C[1], va='top',
                   arrowprops=dict(arrowstyle='->', color=pf.C[1], lw=1))
    right.legend(fontsize=8, loc='lower right', bbox_to_anchor=(1, .02))
    pf.lab(right, 'B')
    return fig
