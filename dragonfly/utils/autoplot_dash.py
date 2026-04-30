#!/usr/bin/env python

'''Dash frontend for Dragonfly autoplot.'''

import argparse
import os
from functools import lru_cache
from urllib.parse import parse_qs, urlencode

import numpy as np

from .autoplot_core import AutoplotController, FrameviewerController, align_models, get_default_normvecs, normalize_highq, subtract_radmin


def _import_dash():
    from dash import Dash, Input, Output, State, dcc, html, ctx
    return Dash, Input, Output, State, dcc, html, ctx


@lru_cache(maxsize=32)
def _load_volume_cached(config_path, fname, modenum, rots, mtime):
    controller = AutoplotController(config=config_path)
    return controller.output_parser.parse_volume(fname, modenum=modenum, rots=rots)


def load_volume(config_path, fname, modenum, rots=True):
    if not fname or not os.path.isfile(fname):
        return None
    mtime = os.path.getmtime(fname)
    return _load_volume_cached(config_path, fname, modenum, rots, mtime)


@lru_cache(maxsize=32)
def _load_log_cached(config_path, logfname, mtime):
    controller = AutoplotController(config=config_path)
    return controller.output_parser.read_log(logfname)


def load_log(config_path, logfname):
    if not logfname or not os.path.isfile(logfname):
        return '', None
    mtime = os.path.getmtime(logfname)
    return _load_log_cached(config_path, logfname, mtime)


@lru_cache(maxsize=8)
def _load_frameviewer_bundle(config_path, mask=False):
    try:
        return FrameviewerController(config_path, mask=mask), None
    except Exception as exc: # pragma: no cover - environment/config dependent
        return None, str(exc)


def load_frameviewer_controller(config_path, mask=False):
    return _load_frameviewer_bundle(config_path, mask=mask)


def build_initial_state(config='config.ini', model=None):
    controller = AutoplotController(config=config, model=model)
    check = controller.check_for_new_iteration(controller.logfname)
    initial_fname = controller.get_initial_volume_fname()
    initial_iteration = 0
    if check['updated'] or check['iteration'] > 0:
        initial_fname = check['fname']
        initial_iteration = check['iteration']
    return {
        'config_path': controller.config,
        'model_name': controller.model_name,
        'logfname': controller.logfname,
        'fname': initial_fname,
        'recon_type': controller.recon_type,
        'num_modes': controller.num_modes,
        'num_nonrot': controller.num_nonrot,
        'num_rot': controller.num_rot,
        'iteration': initial_iteration,
        'max_iternum': controller.max_iternum,
        'layer': None,
        'mode': 0,
        'vrange_min': '0',
        'vrange_max': '1',
        'exponent': '1.0',
        'cmap': 'coolwarm',
        'keep_checking': False,
        'mode_select': False,
        'selected_modes': [],
        'num_good': 0,
        'operations': [],
        'normvecs': get_default_normvecs().tolist(),
        'status': 'Ready',
    }


def apply_saved_preferences(state, preferences):
    if not preferences:
        return state
    config_prefs = preferences.get(state['config_path'], {})
    if not config_prefs:
        return state

    merged = dict(state)
    for key in (
        'cmap',
        'vrange_min',
        'vrange_max',
        'exponent',
        'frameviewer_cmap',
        'frameviewer_plot_max',
    ):
        if key in config_prefs:
            merged[key] = config_prefs[key]
    return merged


def controller_from_state(state):
    controller = AutoplotController(config=state['config_path'], model=state.get('model_name'))
    controller.logfname = state['logfname']
    controller.max_iternum = state.get('max_iternum', 0)
    controller.old_fname = state.get('fname')
    controller.old_modenum = state.get('mode', 0)
    controller.set_mode_selection(state.get('mode_select', False))
    controller.selected_modes = set(state.get('selected_modes', []))
    controller.num_good = state.get('num_good', 0)
    return controller


def build_frameviewer_state(session_state, search=''):
    query = parse_qs((search or '').lstrip('?'))
    mode = int(query.get('mode', [session_state.get('mode', 0)])[0])
    iteration = int(query.get('iteration', [session_state.get('iteration', 0)])[0])
    output_fname = query.get('output', [session_state.get('fname')])[0]
    config_path = query.get('config', [session_state.get('config_path')])[0]
    cmap = query.get('cmap', [session_state.get('cmap', 'coolwarm')])[0]
    controller, error = load_frameviewer_controller(config_path)
    frames = np.array([], dtype='i4') if controller is None else controller.get_available_frames(output_fname=output_fname, mode=mode, skip_bad=False)
    compare_default = True
    compare = query.get('compare', ['1' if compare_default else '0'])[0] not in ('0', 'false', 'False')
    frame_num = int(frames[0]) if len(frames) else 0
    return {
        'config_path': config_path,
        'output_fname': output_fname,
        'iteration': iteration,
        'mode': mode,
        'num_modes': 1 if controller is None else controller.num_modes,
        'compare_vrange_min': query.get('compare_vrange_min', [session_state.get('vrange_min', '0')])[0],
        'compare_vrange_max': query.get('compare_vrange_max', [session_state.get('vrange_max', '1')])[0],
        'compare_exponent': query.get('compare_exponent', [session_state.get('exponent', '1.0')])[0],
        'compare_cmap': query.get('compare_cmap', [session_state.get('cmap', 'coolwarm')])[0],
        'frame_num': frame_num,
        'frame_pos': 0,
        'skip_bad': False,
        'sym': False,
        'compare': compare,
        'plot_max': session_state.get('frameviewer_plot_max', '10'),
        'cmap': session_state.get('frameviewer_cmap', cmap),
        'status': error or 'Frameviewer ready',
    }


def build_frameviewer_href(session_state):
    return '/frameviewer?' + urlencode({
        'config': session_state['config_path'],
        'output': session_state['fname'],
        'iteration': session_state.get('iteration', 0),
        'mode': session_state.get('mode', 0),
        'cmap': session_state.get('cmap', 'coolwarm'),
        'compare_vrange_min': session_state.get('vrange_min', '0'),
        'compare_vrange_max': session_state.get('vrange_max', '1'),
        'compare_exponent': session_state.get('exponent', '1.0'),
        'compare_cmap': session_state.get('cmap', 'coolwarm'),
        'compare': 1,
    })


def apply_operations(parsed, state):
    if parsed is None:
        return None
    vol = parsed['vol']
    for op in state.get('operations', []):
        if op == 'subtract_radmin':
            vol = subtract_radmin(vol, state['recon_type'], state['num_modes'])
        elif op == 'normalize_highq':
            vol = normalize_highq(vol, state['recon_type'])
        elif op == 'align_models':
            vol = align_models(vol, state['recon_type'])
    parsed = dict(parsed)
    parsed['vol'] = vol
    return parsed


def parse_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def make_layout(app, initial_state):
    _, _, _, _, dcc, html, _ = _import_dash()
    control_style = {'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(160px, 1fr))', 'gap': '10px'}
    section_style = {'backgroundColor': '#1b2230', 'padding': '12px', 'borderRadius': '12px', 'marginBottom': '12px', 'border': '1px solid #2e3a50'}
    graph_style = {'backgroundColor': '#131a24', 'padding': '8px', 'borderRadius': '12px', 'border': '1px solid #2e3a50'}
    button_row_style = {'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px', 'alignItems': 'center'}
    initial_frameviewer_state = build_frameviewer_state(initial_state)

    return html.Div([
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='session-state', data=initial_state),
        dcc.Store(id='frameviewer-state', data=initial_frameviewer_state),
        dcc.Store(id='preferences-store', storage_type='local'),
        dcc.Interval(id='poll-interval', interval=5000, disabled=True),
        html.Div([
            html.H2('Dragonfly Autoplot Web', style={'margin': '0 0 6px 0', 'fontSize': '1.4rem'}),
            html.Div(id='status-line', style={'color': '#c8d3f5', 'fontSize': '0.95rem'}),
        ], id='autoplot-page-header', style=section_style),
        html.Div([
        dcc.Tabs([
            dcc.Tab(label='Slices', children=[
                html.Div([
                    html.Div([
                        dcc.Graph(id='mode-grid-graph', config={'displaylogo': False, 'responsive': True}, style={'height': '30vh', 'minHeight': '220px'}),
                    ], id='mode-grid-wrapper', style={**graph_style, 'flex': '1 1 420px'}),
                    html.Div([
                        dcc.Graph(id='main-graph', config={'displaylogo': False, 'responsive': True}, style={'height': '30vh', 'minHeight': '220px'}),
                    ], id='main-graph-wrapper', style={**graph_style, 'flex': '1 1 620px'}),
                ], id='slices-top-row', style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '12px', 'alignItems': 'flex-start'}),
            ], style={'padding': '10px 0'}),
            dcc.Tab(label='Metrics', children=[
                html.Div([
                    dcc.Graph(id='log-graph', config={'displaylogo': False, 'responsive': True}, style={'height': '30vh', 'minHeight': '220px'}),
                ], style=graph_style),
            ], style={'padding': '10px 0'}),
            dcc.Tab(label='Log', children=[
                html.Div([
                    dcc.Textarea(id='log-text', style={'width': '100%', 'height': '100%', 'fontFamily': 'monospace', 'resize': 'none', 'overflowY': 'auto'}),
                ], style={**graph_style, 'height': '30vh', 'minHeight': '220px'}),
            ], style={'padding': '10px 0'}),
        ], colors={'border': '#333', 'primary': '#4f8cff', 'background': '#1a1a1a'}),
        html.Div([
            html.Div([
                html.Div([
                    html.Label('Config file'),
                    dcc.Input(id='config-input', type='text', value=initial_state['config_path'], style={'width': '100%'}),
                ]),
                html.Div([
                    html.Label('Log file'),
                    dcc.Input(id='logfname-input', type='text', value=initial_state['logfname'], style={'width': '100%'}),
                ]),
                html.Div([
                    html.Label('Volume file'),
                    dcc.Input(id='fname-input', type='text', value=initial_state['fname'], style={'width': '100%'}),
                ]),
            ], style={**control_style, 'marginBottom': '10px'}),
            html.Div([
                html.Div([
                    html.Label('Iteration'),
                    dcc.Slider(id='iter-slider', min=0, max=1, step=1, value=initial_state['iteration'], tooltip={'placement': 'bottom'}),
                ], style={'marginBottom': '10px'}),
                html.Div(id='layer-control', children=[
                    html.Label('Layer'),
                    dcc.Slider(id='layer-slider', min=0, max=1, step=1, value=0, tooltip={'placement': 'bottom'}),
                ], style={'marginBottom': '10px'}),
                html.Div(id='mode-control', children=[
                    html.Label('Mode'),
                    dcc.Slider(id='mode-slider', min=0, max=max(initial_state['num_modes'] + initial_state['num_nonrot'] - 1, 0), step=1, value=0, tooltip={'placement': 'bottom'}),
                ]),
            ]),
        ], style=section_style),
        html.Div([
            html.Div([
                html.Div('View', style={'fontWeight': 700, 'marginBottom': '8px', 'color': '#dce6f5'}),
                html.Div([
                    html.Div([
                        html.Label('Color map'),
                        dcc.Dropdown(id='cmap-dropdown', className='dark-dropdown', value=initial_state['cmap'], clearable=False,
                                     options=[{'label': name, 'value': name} for name in ['coolwarm', 'cubehelix', 'CMRmap', 'gray', 'gray_r', 'jet']]),
                    ]),
                    html.Div([
                        html.Label('Range min'),
                        dcc.Input(id='vrange-min-input', type='text', value=initial_state['vrange_min'], debounce=True, style={'width': '100%'}),
                    ]),
                    html.Div([
                        html.Label('Range max'),
                        dcc.Input(id='vrange-max-input', type='text', value=initial_state['vrange_max'], debounce=True, style={'width': '100%'}),
                    ]),
                    html.Div([
                        html.Label('Exponent / log'),
                        dcc.Input(id='exponent-input', type='text', value=initial_state['exponent'], debounce=True, style={'width': '100%'}),
                    ]),
                ], style=control_style),
            ], style={**section_style, 'marginBottom': '0'}),
            html.Div([
                html.Div('Basic Actions', style={'fontWeight': 700, 'marginBottom': '8px', 'color': '#dce6f5'}),
                html.Div([
                    html.Button('Apply Config', id='apply-config-btn'),
                    html.Button('Check', id='check-btn'),
                    html.Button('Plot', id='plot-btn'),
                    html.Button('Reparse', id='reparse-btn'),
                    html.A('Open Frameviewer', id='frameviewer-link', href='/frameviewer', style={'display': 'inline-block', 'padding': '8px 12px', 'backgroundColor': '#1f6feb', 'color': '#f6faff', 'textDecoration': 'none', 'borderRadius': '8px', 'border': '1px solid #397bea'}),
                    dcc.Checklist(id='keep-checking-toggle', options=[{'label': 'Keep checking', 'value': 'enabled'}], value=[]),
                ], style=button_row_style),
            ], style={**section_style, 'marginBottom': '0'}),
            html.Div([
                html.Div('Analysis', style={'fontWeight': 700, 'marginBottom': '8px', 'color': '#dce6f5'}),
                html.Div([
                    html.Button('Subtract Radial Min', id='subtract-btn'),
                    html.Button('Normalize High q', id='normalize-btn'),
                    html.Button('Align Models', id='align-btn'),
                    dcc.Checklist(id='mode-select-toggle', options=[{'label': 'Mode selection', 'value': 'enabled'}], value=[]),
                ], style=button_row_style),
            ], style={**section_style, 'marginBottom': '0'}),
        ], style={'display': 'grid', 'gridTemplateColumns': '1.3fr 1fr 1fr', 'gap': '12px'}),
        ], id='autoplot-page'),
        html.Div([
            html.Div([
                html.H2('Dragonfly Frameviewer', style={'margin': '0 0 6px 0', 'fontSize': '1.4rem'}),
                html.Div(id='frameviewer-status', style={'color': '#c8d3f5', 'fontSize': '0.95rem'}),
            ], style=section_style),
            html.Div([
                html.A('Back to Autoplot', href='/', style={'display': 'inline-block', 'padding': '8px 12px', 'backgroundColor': '#1f6feb', 'color': '#f6faff', 'textDecoration': 'none', 'borderRadius': '8px', 'border': '1px solid #397bea'}),
            ], style={**section_style, 'padding': '10px 12px'}),
            html.Div([
                dcc.Graph(id='frameviewer-graph', config={'displaylogo': False, 'responsive': True}, style={'height': '32vh', 'minHeight': '240px'}),
            ], style=graph_style),
            html.Div([
                html.Div([
                    html.Label('Mode frame index'),
                    dcc.Slider(id='frameviewer-slider', min=0, max=1, step=1, value=0, tooltip={'placement': 'bottom'}),
                    html.Div(id='frameviewer-frame-label', style={'marginTop': '6px', 'color': '#c8d3f5'}),
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Button('Prev', id='frameviewer-prev-btn'),
                    html.Button('Next', id='frameviewer-next-btn'),
                    html.Button('Random', id='frameviewer-random-btn'),
                    dcc.Checklist(id='frameviewer-skipbad-toggle', options=[{'label': 'Skip bad', 'value': 'enabled'}], value=[]),
                    dcc.Checklist(id='frameviewer-sym-toggle', options=[{'label': 'Symmetrize', 'value': 'enabled'}], value=[]),
                    dcc.Checklist(id='frameviewer-compare-toggle', options=[{'label': 'Compare slice', 'value': 'enabled'}], value=[]),
                ], style=button_row_style),
                html.Div([
                    html.Div([
                        html.Label('Mode'),
                        dcc.Slider(id='frameviewer-mode-slider', min=0, max=max(initial_state['num_modes'] - 1, 0), step=1, value=0, tooltip={'placement': 'bottom'}),
                    ]),
                    html.Div([
                        html.Label('Plot max'),
                        dcc.Input(id='frameviewer-plotmax-input', type='text', value='10', debounce=True, style={'width': '100%'}),
                    ]),
                    html.Div([
                        html.Label('Color map'),
                        dcc.Dropdown(id='frameviewer-cmap-dropdown', className='dark-dropdown', value=initial_state['cmap'], clearable=False,
                                     options=[{'label': name, 'value': name} for name in ['coolwarm', 'cubehelix', 'CMRmap', 'gray', 'gray_r', 'jet']]),
                    ]),
                ], style=control_style),
            ], style=section_style),
        ], id='frameviewer-page', style={'display': 'none'}),
    ], style={'maxWidth': '1500px', 'margin': '0 auto', 'padding': '12px', 'backgroundColor': '#0b1118', 'color': '#e6edf7', 'minHeight': '100vh'})


def create_app(config='config.ini', model=None):
    Dash, Input, Output, State, dcc, html, ctx = _import_dash()
    from .autoplot_plotly import build_2d_detail_figure, build_2d_overview_figure, build_3d_figure, build_compare_frame_figure, build_frame_figure, build_log_figure, empty_figure

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.index_string = '''<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body { background: #0b1118; color: #e6edf7; }
            body { margin: 0; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
            label { color: #d9e4f2; font-size: 0.86rem; font-weight: 600; display: block; margin-bottom: 6px; }
            input, textarea, button { font: inherit; }
            input, textarea { background: #101722 !important; color: #e6edf7 !important; border: 1px solid #39485f !important; border-radius: 8px !important; padding: 8px 10px !important; }
            input::placeholder, textarea::placeholder { color: #8fa3bf !important; }
            button { background: #1f6feb; color: #f6faff; border: 1px solid #397bea; border-radius: 8px; padding: 8px 12px; cursor: pointer; }
            button:hover { background: #2b7cff; }
            button.dash-dropdown.dark-dropdown,
            button.dash-dropdown.dark-dropdown:hover,
            button.dash-dropdown.dark-dropdown:focus,
            button.dash-dropdown.dark-dropdown:active {
                background: #101722 !important;
                color: #e6edf7 !important;
                border: 1px solid #39485f !important;
                box-shadow: none !important;
            }
            button.dash-dropdown.dark-dropdown span,
            button.dash-dropdown.dark-dropdown div,
            button.dash-dropdown.dark-dropdown svg,
            button.dash-dropdown.dark-dropdown:hover span,
            button.dash-dropdown.dark-dropdown:focus span {
                background: #101722 !important;
                color: #e6edf7 !important;
                fill: #e6edf7 !important;
            }
            .dark-dropdown .Select-control,
            .dark-dropdown .Select__control,
            .dark-dropdown .select__control,
            .dark-dropdown [class*='-control'] {
                background: #101722 !important; color: #e6edf7 !important; border-color: #39485f !important;
            }

            .Select-menu-outer, .Select-menu,
            .Select__menu, .select__menu, .select__menu-list,
            .dash-dropdown-content,
            .dash-options-list,
            .dash-dropdown-search-container,
            .VirtualizedSelectOption,
            .VirtualizedSelectFocusedOption,
            [class*='menu'] {
                background: #101722 !important;
                color: #e6edf7 !important;
                border-color: #39485f !important;
            }

            .dash-dropdown-search-container,
            .dash-dropdown-search-container > div,
            .dash-dropdown-search,
            .dash-dropdown-search-container svg {
                background: #101722 !important;
                color: #e6edf7 !important;
                border-color: #39485f !important;
            }
            .dash-dropdown-search-container,
            .dash-dropdown-content,
            .dash-dropdown-search-container > div,
            .dash-dropdown-search {
                fill: none !important;
                border: 1px solid #39485f !important;
                outline: 0 !important;
                outline-color: transparent !important;
                outline-style: none !important;
                outline-width: 0 !important;
                box-shadow: none !important;
                -webkit-appearance: none !important;
                appearance: none !important;
            }
            .dash-dropdown-search-container:focus,
            .dash-dropdown-content:focus,
            .dash-dropdown-content:focus-visible,
            .dash-dropdown-search-container:focus-visible,
            .dash-dropdown-search-container > div:focus,
            .dash-dropdown-search-container > div:focus-visible,
            .dash-dropdown-search:focus,
            .dash-dropdown-search:focus-visible {
                outline: 0 !important;
                outline-color: transparent !important;
                box-shadow: none !important;
                border-color: #39485f !important;
            }
            .dash-dropdown-search-container svg {
                fill: #e6edf7 !important;
            }

            .dash-option,
            .Select-option,
            .Select__option,
            .select__option,
            [class*='option'] {
                background: #101722 !important;
                color: #e6edf7 !important;
                border: 0 !important;
                box-shadow: none !important;
            }

            .dash-options-list {
                border: 1px solid #39485f !important;
                box-shadow: none !important;
            }

            .dash-option:hover,
            .Select-option.is-focused,
            .Select-option:hover,
            .Select__option--is-focused,
            .select__option--is-focused,
            .VirtualizedSelectFocusedOption,
            [class*='option']:hover,
            [class*='option'][aria-selected='true'] {
                background: #1b2a3d !important;
                color: #f4f8ff !important;
                border: 0 !important;
                box-shadow: none !important;
            }

            .Select-input, .Select-input > input,
            .Select__input, .Select__input-container,
            .select__input, .select__input-container,
            [class*='input'] {
                color: #e6edf7 !important;
                background: transparent !important;
            }

            .Select-arrow-zone, .Select-arrow,
            .Select-clear-zone,
            .Select-loading-zone,
            .Select-menu-outer svg,
            .Select svg,
            .dash-dropdown svg,
            [class*='indicator'] svg {
                color: #e6edf7 !important;
                fill: #e6edf7 !important;
                background: transparent !important;
            }
            .dark-dropdown .Select-value-label, .dark-dropdown .Select-placeholder, .dark-dropdown .Select-input input,
            .dark-dropdown .VirtualizedSelectOption, .dark-dropdown .VirtualizedSelectFocusedOption,
            .dark-dropdown .Select__single-value, .dark-dropdown .Select__placeholder,
            .dark-dropdown .Select__input-container, .dark-dropdown .Select__option,
            .dark-dropdown .select__single-value, .dark-dropdown .select__placeholder,
            .dark-dropdown .select__input-container, .dark-dropdown .select__option,
            .dark-dropdown .select__option--is-focused, .dark-dropdown .select__option--is-selected,
            .dark-dropdown .select__menu-notice, .dark-dropdown [class*='singleValue'],
            .dark-dropdown [class*='placeholder'], .dark-dropdown [class*='input-container'],
            .dark-dropdown [class*='option'] {
                color: #e6edf7 !important; background: #101722 !important;
            }
            .dark-dropdown .select__indicator svg, .dark-dropdown .Select-arrow-zone,
            .dark-dropdown .Select-arrow, .dark-dropdown [class*='indicator'] svg,
            .dark-dropdown svg {
                color: #e6edf7 !important; fill: #e6edf7 !important;
            }
            .dark-dropdown, .dark-dropdown * { color: #e6edf7 !important; }
            .dark-dropdown input { background: transparent !important; color: #e6edf7 !important; }
            .dash-checkbox label, .dash-checklist label { color: #d9e4f2 !important; }
            .tab { padding: 10px 14px !important; }
            @media (max-width: 900px) {
                #slices-top-row { flex-direction: column; }
                .dash-graph { height: auto !important; }
            }
            @media (max-width: 1200px) {
                body ._dash-app-content > div:last-child { grid-template-columns: 1fr !important; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>'''
    initial_state = build_initial_state(config=config, model=model)
    app.layout = make_layout(app, initial_state)

    @app.callback(
        Output('poll-interval', 'disabled'),
        Input('keep-checking-toggle', 'value'),
    )
    def toggle_interval(values):
        return 'enabled' not in (values or [])

    @app.callback(
        Output('preferences-store', 'data'),
        Input('session-state', 'data'),
        Input('frameviewer-state', 'data'),
        State('preferences-store', 'data'),
        prevent_initial_call=True,
    )
    def sync_preferences(session_state, frameviewer_state, preferences):
        preferences = {} if preferences is None else dict(preferences)
        if session_state is not None:
            config_key = session_state['config_path']
            config_prefs = dict(preferences.get(config_key, {}))
            config_prefs.update({
                'cmap': session_state.get('cmap', 'coolwarm'),
                'vrange_min': session_state.get('vrange_min', '0'),
                'vrange_max': session_state.get('vrange_max', '1'),
                'exponent': session_state.get('exponent', '1.0'),
            })
            preferences[config_key] = config_prefs
        if frameviewer_state is not None:
            config_key = frameviewer_state['config_path']
            config_prefs = dict(preferences.get(config_key, {}))
            config_prefs.update({
                'frameviewer_cmap': frameviewer_state.get('cmap', 'coolwarm'),
                'frameviewer_plot_max': frameviewer_state.get('plot_max', '10'),
            })
            preferences[config_key] = config_prefs
        return preferences

    @app.callback(
        Output('autoplot-page-header', 'style'),
        Output('autoplot-page', 'style'),
        Output('frameviewer-page', 'style'),
        Input('url', 'pathname'),
    )
    def toggle_pages(pathname):
        visible = {'display': 'block'}
        hidden = {'display': 'none'}
        if pathname == '/frameviewer':
            return hidden, hidden, visible
        return visible, visible, hidden

    @app.callback(
        Output('session-state', 'data'),
        Input('apply-config-btn', 'n_clicks'),
        Input('check-btn', 'n_clicks'),
        Input('plot-btn', 'n_clicks'),
        Input('reparse-btn', 'n_clicks'),
        Input('subtract-btn', 'n_clicks'),
        Input('normalize-btn', 'n_clicks'),
        Input('align-btn', 'n_clicks'),
        Input('poll-interval', 'n_intervals'),
        Input('iter-slider', 'value'),
        Input('layer-slider', 'value'),
        Input('mode-slider', 'value'),
        Input('mode-grid-graph', 'clickData'),
        Input('keep-checking-toggle', 'value'),
        Input('mode-select-toggle', 'value'),
        Input('vrange-min-input', 'value'),
        Input('vrange-max-input', 'value'),
        Input('exponent-input', 'value'),
        Input('cmap-dropdown', 'value'),
        State('config-input', 'value'),
        State('logfname-input', 'value'),
        State('fname-input', 'value'),
        State('preferences-store', 'data'),
        State('session-state', 'data'),
        prevent_initial_call=False,
    )
    def update_state(_apply, _check, _plot, _reparse, _subtract, _normalize, _align, _interval,
                     iteration, layer, mode, click_data, keep_checking, mode_select,
                     vrange_min, vrange_max, exponent, cmap,
                     config_path, logfname, fname, preferences, state):
        trigger = ctx.triggered_id
        if state is None or trigger == 'apply-config-btn':
            state = build_initial_state(config=config_path or 'config.ini', model=state.get('model_name') if state else None)
            state = apply_saved_preferences(state, preferences)

        controller = controller_from_state(state)
        state['config_path'] = config_path or state['config_path']
        state['logfname'] = logfname or controller.logfname
        state['fname'] = fname or state['fname']
        state['vrange_min'] = vrange_min
        state['vrange_max'] = vrange_max
        state['exponent'] = exponent
        state['cmap'] = cmap
        state['keep_checking'] = 'enabled' in (keep_checking or [])
        controller.set_mode_selection('enabled' in (mode_select or []))
        controller.selected_modes = set(state.get('selected_modes', []))
        controller.num_good = state.get('num_good', 0)
        state['mode_select'] = controller.mode_select

        if trigger == 'apply-config-btn':
            controller = AutoplotController(config=state['config_path'], model=state.get('model_name'))
            state.update(build_initial_state(config=state['config_path'], model=state.get('model_name')))
            state = apply_saved_preferences(state, preferences)
            state['status'] = 'Loaded config %s' % state['config_path']
            return state

        if iteration is not None and trigger == 'iter-slider':
            state['iteration'] = int(iteration)
            state['fname'] = controller.gen_model_fname(int(iteration))
            state['operations'] = []
        if layer is not None and trigger == 'layer-slider':
            state['layer'] = int(layer)
        if mode is not None and trigger == 'mode-slider':
            state['mode'] = int(mode)

        if trigger in ('reparse-btn', 'iter-slider'):
            state['operations'] = []
        elif trigger == 'subtract-btn':
            state.setdefault('operations', []).append('subtract_radmin')
        elif trigger == 'normalize-btn' and state['recon_type'] == '2d':
            state.setdefault('operations', []).append('normalize_highq')
        elif trigger == 'align-btn' and state['recon_type'] == '2d':
            state.setdefault('operations', []).append('align_models')

        if trigger in ('check-btn', 'poll-interval'):
            update = controller.check_for_new_iteration(state['logfname'])
            state['max_iternum'] = controller.max_iternum
            if update['updated']:
                state['iteration'] = update['iteration']
                state['fname'] = update['fname']
                state['operations'] = []
                state['status'] = 'Detected new iteration %d' % update['iteration']

        if trigger == 'mode-grid-graph' and click_data:
            mode_clicked = click_data['points'][0].get('customdata')
            if mode_clicked is not None:
                mode_clicked = int(mode_clicked)
                parsed = load_volume(state['config_path'], state['fname'], state['mode'], rots=True)
                if parsed is not None and controller.mode_select:
                    controller.toggle_selected_mode(mode_clicked, parsed['modes'])
                    state['selected_modes'] = sorted(controller.selected_modes)
                    state['num_good'] = controller.num_good
                else:
                    state['mode'] = mode_clicked

        state['selected_modes'] = sorted(controller.selected_modes)
        state['num_good'] = controller.num_good
        state['max_iternum'] = max(state.get('max_iternum', 0), controller.max_iternum)
        if trigger in ('vrange-min-input', 'vrange-max-input', 'exponent-input', 'cmap-dropdown'):
            state['status'] = 'Updated view settings'
        elif trigger not in ('check-btn', 'poll-interval', 'apply-config-btn'):
            state['status'] = 'Viewing %s' % os.path.basename(state['fname'])
        return state

    @app.callback(
        Output('status-line', 'children'),
        Output('main-graph', 'figure'),
        Output('mode-grid-graph', 'figure'),
        Output('mode-grid-wrapper', 'style'),
        Output('log-graph', 'figure'),
        Output('log-text', 'value'),
        Output('logfname-input', 'value'),
        Output('fname-input', 'value'),
        Output('vrange-min-input', 'value'),
        Output('vrange-max-input', 'value'),
        Output('exponent-input', 'value'),
        Output('cmap-dropdown', 'value'),
        Output('iter-slider', 'max'),
        Output('iter-slider', 'value'),
        Output('layer-control', 'style'),
        Output('layer-slider', 'max'),
        Output('layer-slider', 'value'),
        Output('mode-control', 'style'),
        Output('mode-slider', 'max'),
        Output('mode-slider', 'value'),
        Output('mode-select-toggle', 'value'),
        Output('frameviewer-link', 'href'),
        Input('session-state', 'data'),
    )
    def render_state(state):
        controller = controller_from_state(state)
        vrange = (parse_float(state['vrange_min'], 0.), parse_float(state['vrange_max'], 1.))
        exponent = state['exponent']
        cmap = state['cmap']
        current_mode = int(state.get('mode', 0))
        parsed = load_volume(state['config_path'], state['fname'], current_mode, rots=True)
        parsed = apply_operations(parsed, state)

        if parsed is None:
            main_fig = empty_figure('Volume file not found or unreadable')
            mode_grid_fig = empty_figure('No mode overview')
            mode_grid_style = {'display': 'none'}
            layer_style = {'display': 'none'}
            mode_style = {'display': 'none'}
            layer_max = 1
            layer_value = 0
            mode_max = max(state['num_modes'] + state['num_nonrot'] - 1, 0)
            mode_value = current_mode
        else:
            if state['recon_type'] == '3d':
                layer_max = parsed['size'] - 1
                layer_state = state.get('layer')
                if layer_state is None:
                    layer_value = parsed['center']
                else:
                    layer_value = min(int(layer_state), layer_max)
                main_fig = build_3d_figure(parsed['vol'], layer_value, vrange, exponent, cmap, np.array(state['normvecs']))
                mode_grid_fig = empty_figure('No mode overview in 3D reconstruction')
                mode_grid_style = {'display': 'none'}
                layer_style = {'backgroundColor': '#1f1f1f', 'padding': '12px', 'borderRadius': '8px', 'marginBottom': '12px'}
                mode_style = {'display': 'none'} if state['num_modes'] <= 1 else {'backgroundColor': '#1f1f1f', 'padding': '12px', 'borderRadius': '8px', 'marginBottom': '12px'}
                mode_max = max(state['num_modes'] - 1, 0)
                mode_value = min(current_mode, mode_max)
            else:
                mode_max = parsed['vol'].shape[0] - 1
                mode_value = min(current_mode, mode_max)
                layer_max = 1
                layer_value = 0
                main_fig = build_2d_detail_figure(parsed['vol'], mode_value, parsed['modes'], vrange, exponent, cmap)
                mode_grid_fig = build_2d_overview_figure(parsed['vol'], state['num_modes'], state['num_nonrot'], set(state['selected_modes']), mode_value, vrange, exponent, cmap)
                mode_grid_style = {'backgroundColor': '#131a24', 'padding': '8px', 'borderRadius': '12px', 'border': '1px solid #2e3a50', 'flex': '1 1 420px'}
                layer_style = {'display': 'none'}
                mode_style = {'backgroundColor': '#1b2230', 'padding': '12px', 'borderRadius': '12px', 'marginBottom': '12px', 'border': '1px solid #2e3a50'} if mode_max > 0 else {'display': 'none'}

        all_lines, loglines = load_log(state['config_path'], state['logfname'])
        o_array = None
        if loglines is not None:
            o_array = controller.output_parser.get_orientations(loglines, np.append(np.where(np.diff(loglines[:, 5].astype('i4')) != 0)[0], loglines[:, 5].shape[0]))
        log_fig = build_log_figure(loglines, o_array, cmap)
        iter_max = max(int(state.get('max_iternum', 0)), 1)
        iter_value = min(int(state.get('iteration', 0)), iter_max)
        status = state.get('status', 'Ready')
        if state.get('mode_select'):
            status += ' | selected modes: %s | good frames: %d' % (state.get('selected_modes', []), state.get('num_good', 0))
        return (
            status,
            main_fig,
            mode_grid_fig,
            mode_grid_style,
            log_fig,
            all_lines,
            state['logfname'],
            state['fname'],
            state['vrange_min'],
            state['vrange_max'],
            state['exponent'],
            state['cmap'],
            iter_max,
            iter_value,
            layer_style,
            layer_max,
            layer_value,
            mode_style,
            mode_max,
            mode_value,
            ['enabled'] if state.get('mode_select') else [],
            build_frameviewer_href(state),
        )

    @app.callback(
        Output('frameviewer-state', 'data'),
        Input('url', 'pathname'),
        Input('url', 'search'),
        Input('frameviewer-prev-btn', 'n_clicks'),
        Input('frameviewer-next-btn', 'n_clicks'),
        Input('frameviewer-random-btn', 'n_clicks'),
        Input('frameviewer-slider', 'value'),
        Input('frameviewer-mode-slider', 'value'),
        Input('frameviewer-skipbad-toggle', 'value'),
        Input('frameviewer-sym-toggle', 'value'),
        Input('frameviewer-compare-toggle', 'value'),
        Input('frameviewer-plotmax-input', 'value'),
        Input('frameviewer-cmap-dropdown', 'value'),
        State('session-state', 'data'),
        State('frameviewer-state', 'data'),
        prevent_initial_call=False,
    )
    def update_frameviewer_state(pathname, search, _prev, _next, _random, slider_value, mode_value,
                                 skip_bad, sym, compare, plot_max, cmap, session_state, fv_state):
        trigger = ctx.triggered_id
        if pathname != '/frameviewer':
            return fv_state
        if fv_state is None or trigger in ('url', None) or trigger == 'url.search' or trigger == 'url.pathname':
            fv_state = build_frameviewer_state(session_state, search)

        fv_state['skip_bad'] = 'enabled' in (skip_bad or [])
        fv_state['sym'] = 'enabled' in (sym or [])
        fv_state['compare'] = 'enabled' in (compare or [])
        fv_state['plot_max'] = plot_max
        fv_state['cmap'] = cmap

        controller, error = load_frameviewer_controller(fv_state['config_path'])
        if controller is None:
            fv_state['status'] = 'Frameviewer unavailable: %s' % error
            return fv_state
        fv_state['num_modes'] = controller.num_modes
        if mode_value is not None and trigger == 'frameviewer-mode-slider':
            fv_state['mode'] = int(mode_value)
            fv_state['frame_pos'] = 0
        frames = controller.get_available_frames(
            output_fname=fv_state['output_fname'],
            mode=fv_state['mode'],
            skip_bad=fv_state['skip_bad'],
        )
        if len(frames) == 0:
            fv_state['frame_num'] = 0
            fv_state['frame_pos'] = 0
            fv_state['status'] = 'No frames available for current filter'
            return fv_state

        curr_pos = min(int(fv_state.get('frame_pos', 0)), len(frames) - 1)
        if trigger == 'frameviewer-slider' and slider_value is not None:
            curr_pos = min(int(slider_value), len(frames) - 1)
        elif trigger == 'frameviewer-prev-btn':
            curr_pos = max(curr_pos - 1, 0)
        elif trigger == 'frameviewer-next-btn':
            curr_pos = min(curr_pos + 1, len(frames) - 1)
        elif trigger == 'frameviewer-random-btn':
            curr_pos = int(np.random.randint(0, len(frames)))
        else:
            if int(fv_state.get('frame_num', frames[curr_pos])) not in set(frames.tolist()):
                curr_pos = 0
            else:
                curr_pos = int(np.where(frames == int(fv_state.get('frame_num', frames[curr_pos])))[0][0])

        fv_state['frame_pos'] = curr_pos
        fv_state['frame_num'] = int(frames[curr_pos])
        fv_state['status'] = 'Frameviewer ready'
        return fv_state

    @app.callback(
        Output('frameviewer-status', 'children'),
        Output('frameviewer-graph', 'figure'),
        Output('frameviewer-slider', 'max'),
        Output('frameviewer-slider', 'value'),
        Output('frameviewer-frame-label', 'children'),
        Output('frameviewer-mode-slider', 'max'),
        Output('frameviewer-mode-slider', 'value'),
        Output('frameviewer-skipbad-toggle', 'value'),
        Output('frameviewer-sym-toggle', 'value'),
        Output('frameviewer-compare-toggle', 'value'),
        Output('frameviewer-plotmax-input', 'value'),
        Output('frameviewer-cmap-dropdown', 'value'),
        Input('frameviewer-state', 'data'),
    )
    def render_frameviewer(fv_state):
        if not fv_state:
            return 'Frameviewer inactive', empty_figure('Open frameviewer from autoplot'), 1, 0, '', 0, 0, [], [], [], '10', 'coolwarm'

        controller, error = load_frameviewer_controller(fv_state['config_path'])
        if controller is None:
            fig = empty_figure('Frameviewer unavailable: %s' % error)
            return 'Frameviewer unavailable', fig, 1, 0, error, max(fv_state.get('num_modes', 1) - 1, 0), fv_state.get('mode', 0), ['enabled'] if fv_state['skip_bad'] else [], ['enabled'] if fv_state['sym'] else [], ['enabled'] if fv_state.get('compare') else [], fv_state['plot_max'], fv_state['cmap']
        frames = controller.get_available_frames(
            output_fname=fv_state['output_fname'],
            mode=fv_state['mode'],
            skip_bad=fv_state['skip_bad'],
        )
        if len(frames) == 0:
            fig = empty_figure('No frames available for current filter')
            return fv_state['status'], fig, 1, 0, 'No frames', max(controller.num_modes - 1, 0), fv_state['mode'], ['enabled'] if fv_state['skip_bad'] else [], ['enabled'] if fv_state['sym'] else [], ['enabled'] if fv_state.get('compare') else [], fv_state['plot_max'], fv_state['cmap']

        pos = min(int(fv_state['frame_pos']), len(frames) - 1)
        frame_num = int(frames[pos])
        frame, cen = controller.get_frame_image(frame_num, sym=fv_state['sym'])
        vmax = parse_float(fv_state['plot_max'], 10.)
        title = controller.get_frame_title(frame, frame_num)
        if fv_state.get('compare'):
            tomo, info = controller.get_compare_slice(fv_state['iteration'], frame_num, sym=fv_state['sym'])
        else:
            tomo, info = None, None
        if tomo is not None:
            compare_vrange = (
                parse_float(fv_state.get('compare_vrange_min'), 0.),
                parse_float(fv_state.get('compare_vrange_max'), 1.),
            )
            fig = build_compare_frame_figure(
                frame, cen, tomo, info, vmax, fv_state['cmap'], title,
                compare_vrange, fv_state.get('compare_exponent', '1.0'), fv_state.get('compare_cmap', 'coolwarm')
            )
        else:
            fig = build_frame_figure(frame, cen, vmax, fv_state['cmap'], title)
        label = 'Mode frame %d of %d visible frames | actual frame %d | mode %d | iteration %d' % (
            pos + 1, len(frames), frame_num, fv_state['mode'], fv_state['iteration'])
        status = fv_state['status']
        if fv_state.get('compare') and tomo is None:
            status = 'Compare unavailable for iteration %d' % fv_state['iteration']
        return status, fig, max(len(frames) - 1, 1), pos, label, max(controller.num_modes - 1, 0), fv_state['mode'], ['enabled'] if fv_state['skip_bad'] else [], ['enabled'] if fv_state['sym'] else [], ['enabled'] if fv_state.get('compare') else [], fv_state['plot_max'], fv_state['cmap']

    return app


def main():
    parser = argparse.ArgumentParser(description='Dragonfly Progress Monitor Web')
    parser.add_argument('-c', '--config_file', help='Path to config file. Default=config.ini', default='config.ini')
    parser.add_argument('-f', '--volume_file', help='Show slices of particular file instead of output', default=None)
    parser.add_argument('--host', help='Dash bind host', default='0.0.0.0')
    parser.add_argument('--port', help='Dash bind port', type=int, default=8050)
    parser.add_argument('--debug', help='Enable Dash debug mode', action='store_true')
    args = parser.parse_args()

    app = create_app(config=args.config_file, model=args.volume_file)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
