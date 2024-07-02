import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_leaflet as dl
import requests
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# Import your simulation function here
from ev_simulation import run_simulation, SimulationParameters, parameter_sweep_chargers

# Custom googlemap script
import googlemap

# Define custom styles
CONTENT_STYLE = {
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

COLUMN_STYLE = {
    "padding": "20px",
    "margin": "0 5px",
    "background-color": "#f8f9fa",
    "border-radius": "5px",
}

# get starting parameters
params = SimulationParameters()

# Create the Dash app
app = dash.Dash(__name__)
    

app.layout = html.Div([
    # Add custom font CSS
    html.Link(
        rel='stylesheet',
        href='https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&family=Lato&family=Roboto+Mono&display=swap'
    ),
    
    html.H1("EV Simulation Dashboard", style={'font-family': 'Roboto, sans-serif', 'font-size': '28px', 'margin-bottom': '20px'}),
    
    html.Div([
        # Input Controls Column (Left Side)
        html.Div([
            html.H2("Input Controls", style={'font-family': 'Roboto, sans-serif', 'font-size': '20px'}),

            dcc.Dropdown(
                id='route-dropdown',
                options=[
                    {'label': 'NYC to Boston: 348km', 'value': 'nyc_boston'},
                    {'label': 'LA to SF: 615km', 'value': 'la_sf'},
                    {'label': 'Chicago to Detroit: 455km', 'value': 'chicago_detroit'}
                ],
                value='nyc_boston'
            ),

            html.Div([
                html.Label("Drivers: ", id='drivers-label', style={'font-family': 'Lato, sans-serif', 'font-size': '16px'}),
                dcc.Slider(
                    id='drivers-slider',
                    min=1, max=1000, value=params.num_drivers, step=1,
                    marks={i: str(i) for i in range(1, 1001, 250)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'margin-top': '20px'}),

            html.Div([
                html.Label("Charging Stations: ", id='stations-label', style={'font-family': 'Lato, sans-serif', 'font-size': '16px', 'padding': '10px 0px'}),
                dcc.Slider(
                    id='stations-slider',
                    min=1, max=100, value=params.num_stations, step=1,
                    marks={i: str(i) for i in range(1, 101, 25)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'margin-top': '20px'}),

            html.Div([
                html.Label("Chargers per Station: ", id='chargers-label', style={'font-family': 'Lato, sans-serif', 'font-size': '16px', 'padding': '10px 0px'}),
                dcc.Slider(
                    id='chargers-slider',
                    min=1, max=12, value=params.num_chargers, step=1,
                    marks={i: str(i) for i in range(1, 13)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'margin-top': '20px'}),

            html.Button('Run Simulation', id='run-button', n_clicks=0, 
                        style={'margin-top': '20px', 'font-family': 'Roboto, sans-serif'}),

            html.Button('Optimize Station Count', id='optimize-button', n_clicks=0, 
                        style={'margin-top': '20px', 'font-family': 'Roboto, sans-serif'}),

            html.Div([
                html.H3("Optimal Station Count", style={'font-family': 'Roboto, sans-serif', 'font-size': '18px'}),
                html.Div(id='optimal-stations', style={'font-family': 'Roboto Mono, monospace', 'font-size': '20px'}),
            ], style={'margin-top': '40px', 'border-top': '1px solid #ccc'}),

            html.Div([
                html.H3("Dead Batteries", style={'font-family': 'Roboto, sans-serif', 'font-size': '18px'}),
                html.Div(id='dead-batteries', style={'font-family': 'Roboto Mono, monospace', 'font-size': '20px', 'color': 'red'}),
            ], style={'margin-top': '40px', 'border-top': '1px solid #ccc'})

        ], style={'width': '18%', 'float': 'left', **COLUMN_STYLE}),
        
        # Right Side (Results and Map)
        html.Div([
            # Results Row
            html.Div([
                html.Div([
                    html.H3("Mileage", style={'font-family': 'Roboto, sans-serif', 'font-size': '18px'}),
                    html.Div(id='avg-mileage', style={'font-family': 'Roboto Mono, monospace', 'font-size': '20px'}),
                ], style={'width': '25%', 'display': 'inline-block'}),
                html.Div([
                    html.H3("Trip Time", style={'font-family': 'Roboto, sans-serif', 'font-size': '18px'}),
                    html.Div(id='avg-trip-time', style={'font-family': 'Roboto Mono, monospace', 'font-size': '20px'}),
                ], style={'width': '25%', 'display': 'inline-block'}),
                html.Div([
                    html.H3("Queue Time", style={'font-family': 'Roboto, sans-serif', 'font-size': '18px'}),
                    html.Div(id='avg-queue-time', style={'font-family': 'Roboto Mono, monospace', 'font-size': '20px'}),
                ], style={'width': '25%', 'display': 'inline-block'}),
                html.Div([
                    html.H3("Charge Time", style={'font-family': 'Roboto, sans-serif', 'font-size': '18px'}),
                    html.Div(id='avg-charging-time', style={'font-family': 'Roboto Mono, monospace', 'font-size': '20px'}),
                ], style={'width': '25%', 'display': 'inline-block'})
            ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),
            
            # Graphs Row
            html.Div([
                dcc.Graph(id='mileage-hist', style={'width': '25%', 'display': 'inline-block'}),
                dcc.Graph(id='trip-time-hist', style={'width': '25%','display': 'inline-block'}),
                dcc.Graph(id='queue-time-hist', style={'width': '25%', 'display': 'inline-block'}),
                dcc.Graph(id='charging-time-hist', style={'width': '25%', 'display': 'inline-block'})  
            ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px', 'height': '25vh'}),
            
            # Map Row
            html.Div([
                html.H2("Route Map", style={'font-family': 'Roboto, sans-serif', 'font-size': '20px'}),
                #dcc.Graph(id='route-map', style={'height': '50vh'})
                #
                html.Div([
                    dl.Map(center=(42.3601, -71.0589), zoom=7, children=[
                        dl.TileLayer(),
                        dl.Polyline(positions=googlemap.points, color='blue')
                    ], style={'width': '100%', 'height': '50vh'})
                ])
                #
            ]),
        ], style={'width': '75%', 'float': 'right', **COLUMN_STYLE}),
    ], style=CONTENT_STYLE),

])

# Update slider labels
@app.callback(
    [Output('drivers-label', 'children'),
     Output('stations-label', 'children'),
     Output('chargers-label', 'children')],
    [Input('drivers-slider', 'value'),
     Input('stations-slider', 'value'),
     Input('chargers-slider', 'value')]
)

def update_slider_labels(drivers, stations, chargers):
    return f"Drivers: {drivers}", f"Charging Stations: {stations}", f"Chargers per Station: {chargers}"


# callback functions
@app.callback(
    [Output('avg-trip-time', 'children'),
     Output('avg-mileage', 'children'),
     Output('avg-queue-time', 'children'),
     Output('avg-charging-time', 'children'),
     Output('dead-batteries', 'children'),
     Output('trip-time-hist', 'figure'),
     Output('mileage-hist', 'figure'),
     Output('queue-time-hist', 'figure'),
     Output('charging-time-hist', 'figure'),
    #  Output('route-map', 'figure'),
     Output('optimal-stations', 'children'),
     Output('stations-slider', 'value')],
    [Input('run-button', 'n_clicks'),
     Input('optimize-button', 'n_clicks')],
    [State('route-dropdown', 'value'),
     State('drivers-slider', 'value'),
     State('stations-slider', 'value'),
     State('chargers-slider', 'value')],
    prevent_initial_call=True
)

def update_output(n_clicks_run, n_clicks_optimize, route, num_drivers, num_stations, num_chargers):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    params = SimulationParameters()
    params.num_drivers = num_drivers
    params.num_stations = num_stations
    params.num_chargers = num_chargers
    
    if route == 'nyc_boston':
        params.loc_destination = 348  # km
    elif route == 'la_sf':
        params.loc_destination = 615  # km 
    elif route == 'chicago_detroit':
        params.loc_destination = 455  # km 
    
    if button_id == 'optimize-button':
        optimal_result = parameter_sweep_chargers(params)
        if optimal_result:
            results = optimal_result['results']
            optimal_stations = optimal_result['optimal_stations']
            # !! do we need to run_simulation again with optimal_stations?
        else:
            results = run_simulation(params)
            optimal_stations = num_stations
            # !! add a message to the user that optimization failed 
    else:
        results_df, results_dict = run_simulation(params)
        optimal_stations = num_stations

    trip_time_hist = px.histogram(x=results_df['trip_time'], nbins=20)
    mileage_hist = px.histogram(x=results_df['mileage'], nbins=20)
    queue_time_hist = px.histogram(x=results_df['queue_time'], nbins=20)
    charging_time_hist = px.histogram(x=results_df['charge_time'], nbins=20)
    
    
    # # Dummy map for now
    # map_fig = go.Figure(go.Scattermapbox(
    #     mode = "markers+lines",
    #     lon = [-74.0060, -71.0589],
    #     lat = [40.7128, 42.3601],
    #     marker = {'size': 10}))

    # map_fig.update_layout(
    #     mapbox_style="open-street-map",
    #     mapbox=dict(
    #         center=dict(lon=-72.5, lat=41.5),
    #         zoom=6
    #     )
    # )
    
    return (
        f"Avg: {results_dict['avg_trip_time']:.1f} min",
        f"Avg: {results_dict['avg_mileage']:.1f} km",
        f"Avg: {results_dict['avg_queue_time']:.1f} min",
        f"Avg: {results_dict['avg_charging_time']:.1f} min",
        f"Count: {results_dict['dead_batteries']}",
        trip_time_hist,
        mileage_hist,
        queue_time_hist,
        charging_time_hist,
        # map_fig,
        f"Optimal: {optimal_stations}",
        optimal_stations
    )


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
