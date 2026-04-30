'''Plotly figure builders for the autoplot web frontend.'''

import numpy as np
import matplotlib.cm
import matplotlib.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .autoplot_core import get_mode_grid_shape, get_normslice


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref='paper', yref='paper')
    fig.update_layout(template='plotly_dark', margin=dict(l=20, r=20, t=40, b=20))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def mpl_cmap_to_plotly(cmap_name, n=256):
    cmap = matplotlib.cm.get_cmap(cmap_name, n)
    return [(i / float(n - 1), matplotlib.colors.to_hex(cmap(i))) for i in range(n)]


def transform_image(image, vrange, exponent):
    arr = np.array(image, dtype='f8', copy=True)
    raw = arr.copy()
    arr[~np.isfinite(arr)] = np.nan
    rangemin, rangemax = vrange
    if rangemax <= rangemin:
        rangemax = rangemin + 1.

    clipped = np.clip(arr, rangemin, rangemax)
    if exponent == 'log':
        linthresh = max(abs(rangemax) * 1.e-2, 1.e-12)

        def _symlog(val):
            return np.sign(val) * np.log10(1. + np.abs(val) / linthresh)

        disp = _symlog(clipped)
        zmin = _symlog(rangemin)
        zmax = _symlog(rangemax)
    else:
        gamma = float(exponent)
        normed = (clipped - rangemin) / (rangemax - rangemin)
        normed = np.clip(normed, 0., 1.)
        disp = np.power(normed, gamma)
        zmin = 0.
        zmax = 1.

    return disp, raw, zmin, zmax


def build_3d_figure(vol, layer, vrange, exponent, cmap, normvecs):
    fig = make_subplots(rows=1, cols=3, subplot_titles=[str(np.round(vec, 3)) for vec in normvecs], horizontal_spacing=0.03)
    colorscale = mpl_cmap_to_plotly(cmap)
    for index, vec in enumerate(normvecs):
        vslice = get_normslice(vol, vec, layer)
        disp, raw, zmin, zmax = transform_image(vslice, vrange, exponent)
        fig.add_trace(
            go.Heatmap(
                z=disp,
                customdata=raw,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                showscale=(index == 0),
                colorbar=dict(title='Intensity', len=0.85, y=0.5),
                hovertemplate='x=%{x}<br>y=%{y}<br>I=%{customdata}<extra></extra>',
            ),
            row=1,
            col=index+1,
        )
        fig.update_xaxes(visible=False, row=1, col=index+1)
        fig.update_yaxes(visible=False, autorange='reversed', scaleanchor=f'x{index+1}', row=1, col=index+1)

    fig.update_layout(template='plotly_dark', margin=dict(l=10, r=10, t=40, b=10), height=420)
    return fig


def build_2d_detail_figure(vol, mode, modes, vrange, exponent, cmap):
    disp, raw, zmin, zmax = transform_image(vol[mode], vrange, exponent)
    title = 'Class %d' % mode
    if modes is not None:
        title = 'Class %d (%d frames)' % (mode, int((modes == mode).sum()))
    fig = go.Figure(
        go.Heatmap(
            z=disp,
            customdata=raw,
            colorscale=mpl_cmap_to_plotly(cmap),
            zmin=zmin,
            zmax=zmax,
            hovertemplate='x=%{x}<br>y=%{y}<br>I=%{customdata}<extra></extra>',
            colorbar=dict(title='Intensity'),
        )
    )
    fig.update_layout(
        template='plotly_dark',
        title=dict(text=title, x=0.5, xanchor='center'),
        margin=dict(l=10, r=10, t=44, b=10),
        height=420,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange='reversed', scaleanchor='x')
    return fig


def build_2d_overview_figure(vol, num_modes, num_nonrot, selected_modes, current_mode, vrange, exponent, cmap):
    numx, numy = get_mode_grid_shape(num_modes, num_nonrot)
    total = num_modes + num_nonrot
    titles = []
    for mode in range(total):
        label = str(mode)
        if mode == current_mode:
            label = '[%s]' % label
        if mode in selected_modes:
            label = '*%s*' % label
        titles.append(label)
    fig = make_subplots(rows=numy, cols=numx, subplot_titles=titles, horizontal_spacing=0.03, vertical_spacing=0.08)
    colorscale = mpl_cmap_to_plotly(cmap)
    for mode in range(total):
        row = mode // numx + 1
        col = mode % numx + 1
        disp, raw, zmin, zmax = transform_image(vol[mode], vrange, exponent)
        fig.add_trace(
            go.Heatmap(
                z=disp,
                customdata=np.full_like(raw, mode, dtype='i4'),
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                showscale=False,
                hovertemplate='Mode %{customdata}<extra></extra>',
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, autorange='reversed', scaleanchor=f'x{mode+1}', row=row, col=col)
    fig.update_layout(template='plotly_dark', margin=dict(l=10, r=10, t=50, b=10), height=max(280, 165 * numy))
    return fig


def build_log_figure(loglines, o_array, cmap):
    if loglines is None or len(loglines) == 0:
        return empty_figure('No log iterations parsed yet')

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{'rowspan': 2}, {}, {'rowspan': 2}], [None, {}, None]],
        subplot_titles=['RMS change', r'Mutual info. I(K,Omega | W)', 'Most likely orientations', 'Avg log-likelihood'],
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
        column_widths=[0.29, 0.29, 0.42],
        row_heights=[0.5, 0.5],
    )

    iternum = loglines[:, 0].astype('i4')
    num_rot = loglines[:, 5].astype('i4')
    beta = loglines[:, 6].astype('f8')
    num_rot_change = np.append(np.where(np.diff(num_rot) != 0)[0], num_rot.shape[0])
    beta_change = np.where(np.diff(beta) != 0.)[0]

    fig.add_trace(go.Scatter(x=iternum, y=loglines[:, 2].astype('f8'), mode='lines+markers', name='RMS change'), row=1, col=1)
    fig.add_trace(go.Scatter(x=iternum, y=loglines[:, 3].astype('f8'), mode='lines+markers', name='Mutual info'), row=1, col=2)
    if len(iternum) > 1:
        fig.add_trace(go.Scatter(x=iternum[1:], y=loglines[1:, 4].astype('f8'), mode='lines+markers', name='Avg log-likelihood'), row=2, col=2)

    for change in beta_change:
        xval = change + 1 - 0.1
        fig.add_vline(x=xval, line_dash='dash', line_color='white', row=1, col=1)
        fig.add_vline(x=xval, line_dash='dash', line_color='white', row=1, col=2)
        fig.add_vline(x=xval, line_dash='dash', line_color='white', row=2, col=2)
    for change in num_rot_change[:-1]:
        xval = change + 1 + 0.1
        fig.add_vline(x=xval, line_dash='dash', line_color='orange', row=1, col=1)
        fig.add_vline(x=xval, line_dash='dash', line_color='orange', row=1, col=2)
        fig.add_vline(x=xval, line_dash='dash', line_color='orange', row=2, col=2)

    if o_array is not None and o_array.shape[1] > 1:
        filtered = o_array[o_array[:, -1] >= 0]
        fig.add_trace(
            go.Heatmap(z=filtered, colorscale=mpl_cmap_to_plotly(cmap), showscale=False, hovertemplate='Orientation=%{z}<extra></extra>'),
            row=1,
            col=3,
        )
        fig.update_yaxes(showticklabels=False, row=1, col=3)

    fig.update_layout(template='plotly_dark', margin=dict(l=10, r=10, t=50, b=10), height=420, showlegend=False)
    fig.update_xaxes(title='Iteration', row=1, col=1)
    fig.update_xaxes(title='Iteration', row=1, col=2)
    fig.update_xaxes(title='Iteration', row=2, col=2)
    fig.update_xaxes(title='Iteration', row=1, col=3)
    return fig


def build_frame_figure(frame, cen, vmax, cmap, title):
    fig = go.Figure(
        go.Heatmap(
            z=np.asarray(frame).T,
            colorscale=mpl_cmap_to_plotly(cmap),
            zmin=0,
            zmax=vmax,
            hovertemplate='x=%{x}<br>y=%{y}<br>I=%{z}<extra></extra>',
            colorbar=dict(title='Photons'),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[cen[0]],
            y=[cen[1]],
            mode='markers',
            marker=dict(symbol='cross', color='lime', size=14),
            hoverinfo='skip',
            showlegend=False,
        )
    )
    fig.update_layout(
        template='plotly_dark',
        title=dict(text=title, x=0.5, xanchor='center'),
        margin=dict(l=10, r=10, t=42, b=10),
        height=440,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange='reversed', scaleanchor='x')
    return fig


def build_compare_frame_figure(frame, cen, tomo, info, vmax, cmap, title, compare_vrange, compare_exponent, compare_cmap):
    fig = make_subplots(rows=1, cols=2, subplot_titles=[title, 'Mutual Info. = %f' % info], horizontal_spacing=0.05)
    colorscale = mpl_cmap_to_plotly(cmap)
    tomo_disp, tomo_raw, tomo_zmin, tomo_zmax = transform_image(tomo, compare_vrange, compare_exponent)
    fig.add_trace(
        go.Heatmap(
            z=np.asarray(frame).T,
            colorscale=colorscale,
            zmin=0,
            zmax=vmax,
            hovertemplate='x=%{x}<br>y=%{y}<br>I=%{z}<extra></extra>',
            showscale=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[cen[0]],
            y=[cen[1]],
            mode='markers',
            marker=dict(symbol='cross', color='lime', size=14),
            hoverinfo='skip',
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=tomo_disp.T,
            colorscale=mpl_cmap_to_plotly(compare_cmap),
            zmin=tomo_zmin,
            zmax=tomo_zmax,
            customdata=tomo_raw.T,
            hovertemplate='x=%{x}<br>y=%{y}<br>I=%{customdata}<extra></extra>',
            colorbar=dict(title='Intensity'),
        ),
        row=1,
        col=2,
    )
    fig.update_layout(template='plotly_dark', margin=dict(l=10, r=10, t=42, b=10), height=440)
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, autorange='reversed', scaleanchor='x', row=1, col=1)
    fig.update_yaxes(visible=False, autorange='reversed', scaleanchor='x2', row=1, col=2)
    return fig
