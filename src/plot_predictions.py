#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot predictions from the text output of predict.py.

Reads one or more files (or stdin) containing predict.py stdout, parses the
Predicted and Observed columns (pKa output with --show-label), and produces
a hexbin or scatter plot of predicted vs reference with optional regression
and score annotation.

Usage:
  python plot_predictions.py predict_output.txt [options]
  python predict.py --pka --atomic --numpy --show-label ... | python plot_predictions.py -
"""

import argparse
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np


# -----------------------------------------------------------------------------
# Score and utility functions
# -----------------------------------------------------------------------------

def rmsd(x, y):
    """Root-mean-square deviation between x and y."""
    return np.sqrt(np.mean((np.asarray(x) - np.asarray(y)) ** 2))


def pearson_r(x, y):
    """Pearson correlation coefficient."""
    x, y = np.asarray(x), np.asarray(y)
    return np.corrcoef(x, y)[0, 1]


def mape(x, y):
    """Mean absolute percentage error (as decimal)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = x != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y[mask] - x[mask]) / x[mask]))


def generate_ticks(lo, hi, n_ticks):
    """n_ticks evenly spaced tick values in [lo, hi]."""
    return np.linspace(lo, hi, n_ticks).tolist()


# -----------------------------------------------------------------------------
# Parsing predict.py output
# -----------------------------------------------------------------------------

def _is_header(line):
    """True if line looks like a predict.py header (contains Predicted or ΔG)."""
    return bool(re.search(r'Predicted|ΔG|Observed', line))


def _parse_row(line, pred_idx, obs_idx):
    """Parse one data row; return (pred, obs) or (pred, None)."""
    parts = line.split()
    if len(parts) <= max(pred_idx, obs_idx if obs_idx is not None else 0):
        return None, None
    try:
        pred = float(parts[pred_idx])
        obs = float(parts[obs_idx]) if obs_idx is not None else None
        return pred, obs
    except (ValueError, IndexError):
        return None, None


def parse_predict_output(lines):
    """
    Parse predict.py text output into reference (x) and predicted (y) arrays.

    Expects first line to be a header with "Predicted" and optionally "Observed".
    Data rows are space-separated; first column is Predicted, second is Observed
    when present (pKa with --show-label).

    Parameters
    ----------
    lines : iterable of str
        Lines from a predict.py output file or stdout.

    Returns
    -------
    x : np.ndarray
        Reference/observed values (y-axis convention: reference on x).
    y : np.ndarray
        Predicted values.

    Raises
    ------
    ValueError
        If no valid data rows or no Observed column for 2D plot.
    """
    lines = list(lines)
    if not lines:
        raise ValueError('Empty input.')

    # Detect header and column indices
    header = lines[0].strip()
    pred_idx = 0
    obs_idx = None
    if 'Observed' in header:
        # "Predicted  Observed  AA ..." -> obs_idx = 1
        obs_idx = 1
    # If no Observed, we only have predicted (single column for pred)

    x_list, y_list = [], []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        pred, obs = _parse_row(line, pred_idx, obs_idx)
        if pred is None:
            continue
        y_list.append(pred)
        x_list.append(obs if obs is not None else len(y_list) - 1)

    y = np.array(y_list, dtype=float)
    x = np.array(x_list, dtype=float)
    if len(y) == 0:
        raise ValueError('No valid data rows found in predict.py output.')
    has_observed = obs_idx is not None
    if not has_observed:
        x = np.arange(len(y), dtype=float)
    return x, y, has_observed


def load_data(file_or_path, clean=False, clean_threshold=-700):
    """
    Load (x, y) from a file path, file handle, or '-' for stdin.
    If clean=True, keep only rows where x >= clean_threshold.
    """
    if file_or_path == '-' or file_or_path is None:
        lines = sys.stdin.readlines()
    elif hasattr(file_or_path, 'readlines'):
        lines = file_or_path.readlines()
    else:
        with open(file_or_path, 'r') as f:
            lines = f.readlines()

    x, y, has_observed = parse_predict_output(lines)
    if clean:
        mask = x >= clean_threshold
        x, y = x[mask], y[mask]
    if not has_observed:
        sys.stderr.write('Warning: No Observed column in input. Run predict.py with --show-label for predicted vs reference.\n')
    return x, y, has_observed


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_preds(
    x,
    y,
    *,
    units=None,
    log=True,
    save=False,
    div=1,
    add=0,
    linreg=True,
    subplot=None,
    title=None,
    xlabel='reference value',
    ylabel='predicted value',
    hexbin=False,
    annotate=False,
    size=5,
    dpi=600,
    color='green',
    colors=None,
    dx=None,
    dy=None,
    show_r=True,
    multi_r=False,
    score_mape=False,
    fontsize=18,
    pad=None,
    wspace=None,
    hspace=None,
    range_limit=False,
    a_size=8,
    eq_axes=False,
    n_ticks=5,
    title_weight='bold',
    no_b=True,
    legend_loc='upper left',
    label_fontsize=13,
    framealpha=0.2,
    print_shape=False,
):
    """
    Plot predicted (y) vs reference (x) with optional hexbin, regression, and scores.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if div != 1 or add != 0:
        x = x / div + add
        y = y / div + add

    if print_shape:
        print('x -', x.shape)
        print('y -', y.shape)

    score_fn = mape if score_mape else pearson_r
    score_label = 'MAPE' if score_mape else 'r'

    if subplot is None:
        plt.clf()
    else:
        plt.subplot(*subplot)

    # Axis range
    if range_limit is not False and range_limit is not None:
        lo, hi = 0, float(range_limit)
    else:
        lo = min(np.min(x), np.min(y)) - 1
        hi = max(np.max(x), np.max(y)) + 1
    t = np.linspace(lo - 100, hi + 100)
    common_range = [min(lo, hi), max(lo, hi)]
    plt.xlim(common_range)
    plt.ylim(common_range)

    if eq_axes:
        ticks = generate_ticks(common_range[0], common_range[1], n_ticks)
        plt.gca().set_xticks(ticks)
        plt.gca().set_yticks(ticks)

    plt.gca().tick_params(axis='x', labelsize=label_fontsize)
    plt.gca().tick_params(axis='y', labelsize=label_fontsize)
    plt.gca().set_aspect('equal', adjustable='box')

    # Title
    if title is not None and title != '':
        plt.title(title, fontsize=fontsize, pad=pad, fontweight=title_weight)
    elif title == '':
        pass  # blank title by default
    else:
        rmse_str = f'RMSE: {rmsd(x, y):.2f}'
        if units:
            rmse_str += f' {units}'
        plt.title(rmse_str, fontsize=fontsize, pad=pad)

    # Axis labels
    xl = xlabel or ('Reference Value' + (f' [{units}]' if units else ''))
    yl = ylabel or ('Predicted Value' + (f' [{units}]' if units else ''))
    plt.xlabel(xl, fontsize=label_fontsize)
    plt.ylabel(yl, fontsize=label_fontsize)

    # Main plot: hexbin or scatter
    if hexbin:
        if log:
            hb = plt.hexbin(x, y, gridsize=100, cmap='gnuplot2', norm=matplotlib.colors.LogNorm())
        else:
            hb = plt.hexbin(x, y, gridsize=100, cmap='gnuplot2')
        plt.colorbar(hb, label='Frequency')
    else:
        if dx is None:
            dx = np.zeros_like(x)
        if dy is None:
            dy = np.zeros_like(y)
        if np.any(dx != 0) or np.any(dy != 0):
            plt.errorbar(x, y, xerr=dx, yerr=dy, fmt='none', ecolor='gray', elinewidth=1, capsize=3, zorder=1)
        if colors is None:
            lbl = f'RMSE: {rmsd(x, y):.2f}' if show_r and not multi_r else None
            plt.scatter(x, y, s=size, c=color, label=lbl)
        else:
            for item in colors:
                if len(item) == 4:
                    lbl, c, s, mask = item
                    if multi_r:
                        lbl = f'{score_label} = {score_fn(x[mask], y[mask]):.3f}'
                    plt.scatter(x[mask], y[mask], s=s, c=c, label=lbl)
                else:
                    lbl, c, s = item
                    if multi_r:
                        lbl = f'{score_label} = {score_fn(x, y):.3f}'
                    plt.scatter(x, y, s=s, c=c, label=lbl)
        if annotate:
            for i in range(len(x)):
                plt.annotate(str(i), (x[i], y[i]), fontsize=a_size)

    # 1:1 line and score
    if linreg:
        coef = np.polyfit(x, y, 1)
        poly_fn = np.poly1d(coef)
        lbl = f'm={coef[0]:.2f}'
        if not no_b:
            lbl += f', b={coef[1]:.2f}'
        plt.plot(x, poly_fn(x), '--k', label=lbl)

    if show_r and not multi_r and colors is None:
        plt.plot(t, t, ls='dashed', linewidth=0.5, color='black', label=f'{score_label} = {score_fn(x, y):.3f}')
    else:
        plt.plot(t, t, ls='dashed', linewidth=0.5, color='black')

    plt.legend(loc=legend_loc, fontsize=label_fontsize, framealpha=framealpha)

    if wspace is not None or hspace is not None:
        plt.subplots_adjust(wspace=wspace, hspace=hspace)

    if save:
        plt.savefig(save, dpi=dpi)
        print('Plot saved to', save)
    elif subplot is None:
        plt.show()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Plot predictions from predict.py output (predicted vs observed).',
        epilog='Input: path to file(s) with predict.py stdout, or "-" for stdin. Multiple files are concatenated.',
    )
    p.add_argument('input', nargs='*', default=['-'], help='Output file(s) from predict.py, or "-" for stdin; default: stdin when no args')
    # Data
    p.add_argument('--clean', action='store_true', help='Drop rows where reference (x) < -700')
    p.add_argument('--units', default=None, help='Units string for axis labels and title')
    p.add_argument('--div', type=float, default=1, help='Divide x,y by this before plotting')
    p.add_argument('--add', type=float, default=0, help='Add this to x,y after div')
    # Plot type
    p.add_argument('--hexbin', action='store_true', help='Use hexbin instead of scatter (default: scatter)')
    p.add_argument('--log', action='store_true', default=True, help='Log scale for hexbin count (default: True)')
    p.add_argument('--no-log', action='store_false', dest='log', help='Linear hexbin count')
    p.add_argument('--size', type=float, default=5, help='Scatter point size')
    p.add_argument('--color', default='green', help='Scatter color (default: green)')
    # Regression and score
    p.add_argument('--linreg', action='store_true', default=True, help='Plot linear regression line (default)')
    p.add_argument('--no-linreg', action='store_false', dest='linreg', help='Do not plot regression line')
    p.add_argument('--no-b', action='store_true', help='Omit intercept from regression label (default)')
    p.add_argument('--intercept', action='store_true', dest='show_b', help='Include intercept in regression label')
    p.add_argument('--show-r', action='store_true', default=True, help='Show correlation/MAPE on 1:1 line (default)')
    p.add_argument('--no-show-r', action='store_false', dest='show_r')
    p.add_argument('--mape', action='store_true', dest='score_mape', help='Use MAPE instead of Pearson r')
    p.add_argument('--multi-r', action='store_true', help='Per-series score when using colors')
    # Labels and title
    p.add_argument('--title', default='', help='Plot title (default: blank)')
    p.add_argument('--xlabel', default='reference value', help='X-axis label')
    p.add_argument('--ylabel', default='predicted value', help='Y-axis label')
    p.add_argument('--annotate', action='store_true', help='Annotate points with index')
    p.add_argument('--a-size', type=float, default=8, help='Annotation font size')
    # Axes
    p.add_argument('--range-limit', type=float, default=None, metavar='HI', help='Set axis max (min=0)')
    p.add_argument('--eq-axes', action='store_true', help='Equal tick spacing on x and y')
    p.add_argument('--n-ticks', type=int, default=5, help='Number of ticks when --eq-axes')
    # Fonts and layout
    p.add_argument('--fontsize', type=float, default=18, help='Title font size')
    p.add_argument('--label-fontsize', type=float, default=13, help='Axis and legend font size')
    p.add_argument('--pad', type=float, default=None, help='Title pad')
    p.add_argument('--title-weight', default='bold', help='Title font weight')
    p.add_argument('--legend-loc', default='upper left', help='Legend location')
    p.add_argument('--framealpha', type=float, default=0.2, help='Legend frame alpha')
    p.add_argument('--wspace', type=float, default=None, help='Subplots wspace')
    p.add_argument('--hspace', type=float, default=None, help='Subplots hspace')
    # Output
    p.add_argument('--save', '-o', default=None, metavar='PATH', help='Save figure to this path (default: predictions.png)')
    p.add_argument('--show', action='store_true', help='Show figure instead of saving (use when a GUI is available)')
    p.add_argument('--dpi', type=int, default=600, help='DPI for saved figure')
    p.add_argument('--print-shape', action='store_true', help='Print x,y shape')
    # Subplot (for multi-panel; single plot by default)
    p.add_argument('--subplot', type=int, nargs=3, default=None, metavar=('NROWS', 'NCOLS', 'INDEX'), help='Subplot position')
    # Uncertainty (scatter mode)
    p.add_argument('--dx', type=float, nargs='+', default=None, help='X errors for errorbar')
    p.add_argument('--dy', type=float, nargs='+', default=None, help='Y errors for errorbar')
    return p.parse_args()


def main():
    args = parse_args()
    inputs = args.input
    if inputs == ['-']:
        inputs = [None]

    all_x, all_y = [], []
    for inp in inputs:
        x, y, _ = load_data(inp, clean=args.clean)
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())

    x = np.array(all_x)
    y = np.array(all_y)
    if len(x) == 0:
        print('No data to plot.', file=sys.stderr)
        sys.exit(1)

    dx = np.array(args.dx) if args.dx is not None else None
    dy = np.array(args.dy) if args.dy is not None else None

    # Default: save to predictions.png; use --show to display instead
    save_path = None if args.show else (args.save or 'predictions.png')

    plot_preds(
        x,
        y,
        units=args.units,
        log=args.log,
        save=save_path,
        div=args.div,
        add=args.add,
        linreg=args.linreg,
        subplot=tuple(args.subplot) if args.subplot else None,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        hexbin=args.hexbin,
        annotate=args.annotate,
        size=args.size,
        dpi=args.dpi,
        color=args.color,
        colors=None,
        dx=dx,
        dy=dy,
        show_r=args.show_r,
        multi_r=args.multi_r,
        score_mape=args.score_mape,
        fontsize=args.fontsize,
        pad=args.pad,
        wspace=args.wspace,
        hspace=args.hspace,
        range_limit=args.range_limit,
        a_size=args.a_size,
        eq_axes=args.eq_axes,
        n_ticks=args.n_ticks,
        title_weight=args.title_weight,
        no_b=not getattr(args, 'show_b', False),
        legend_loc=args.legend_loc,
        label_fontsize=args.label_fontsize,
        framealpha=args.framealpha,
        print_shape=args.print_shape,
    )


if __name__ == '__main__':
    main()
