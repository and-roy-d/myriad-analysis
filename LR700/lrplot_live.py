import plotly.graph_objects as go
import time
import numpy as np
import pathlib
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import datetime
import pytz

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id='live-graph-rt', config={'displayModeBar': False}, style={'height': '60vh'})
    ], style={'width': '100%', 'display': 'inline-block'}),

    html.Div([
        html.Div([
            dcc.Graph(id='live-graph-r', config={'displayModeBar': False}, style={'height': '30vh'})
        ], style={'width': '50%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(id='live-graph-t', config={'displayModeBar': False}, style={'height': '30vh'})
        ], style={'width': '50%', 'display': 'inline-block'})
    ], style={'width': '100%', 'display': 'inline-block'}),
    dcc.Interval(
        id='interval-component',
        interval=5*1000,
        n_intervals=0
    )
])

@app.callback(
    [Output('live-graph-rt', 'figure'),
     Output('live-graph-r', 'figure'),
     Output('live-graph-t', 'figure')],
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    try:
        path = pathlib.Path(__file__)
        dirs = list((path.parent / "Data").glob("*"))
        dir_ = dirs[-1]
        files = list(dir_.glob("*"))
        file = files[-1]
        data = np.load(file)

        num_points = len(data["times_s"])
        last_5_indices = list(range(max(0, num_points - 5), num_points))

        plot_color = 'orange'
        current_color = 'red'

        utc_times = [datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc) for ts in data["times_s"]]
        mst_timezone = pytz.timezone("America/Denver")
        mst_times = [utc_time.astimezone(mst_timezone) for utc_time in utc_times]

        fig_rt = go.Figure()
        fig_rt.add_trace(go.Scatter(
            x=data["t_K"]*1000,
            y=data["r_ohm"] * 1000,
            mode='markers',
            name='R vs T',
            marker=dict(size=4, color=[plot_color]*(num_points-len(last_5_indices))+[current_color]*len(last_5_indices))
        ))
        fig_rt.update_layout(title="R vs T", xaxis_title="Temperature (mK)", yaxis_title="Resistance (mΩ)", margin=dict(l=50, r=50, t=50, b=50), template="plotly_white")

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(
            x=mst_times,
            y=data["r_ohm"] * 1000,
            mode='markers',
            name='Resistance',
            marker=dict(size=4, color=[plot_color]*(num_points-len(last_5_indices))+[current_color]*len(last_5_indices))
        ))
        fig_r.update_layout(title="Live Resistance Data", xaxis_title="Local Time (MST)", yaxis_title="Resistance (mΩ)", margin=dict(l=50, r=50, t=50, b=50), template="plotly_white")

        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=mst_times,
            y=data["t_K"] * 1000,
            mode='markers',
            name='Temperature',
            marker=dict(size=4, color=[plot_color]*(num_points-len(last_5_indices))+[current_color]*len(last_5_indices))
        ))
        fig_t.update_layout(title="Live Temperature Data", xaxis_title="Local Time (MST)", yaxis_title="Temperature (mK)", margin=dict(l=50, r=50, t=50, b=50), template="plotly_white")

        return fig_rt, fig_r, fig_t

    except (FileNotFoundError, IndexError):
        print(f"Warning: No data found. Skipping update.")
        return go.Figure(), go.Figure(), go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)