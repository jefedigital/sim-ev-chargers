import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# Import your simulation function here
from ev_simulation import run_simulation, SimulationParameters

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

# Create the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    # Add custom font CSS
    html.Link(
        rel='stylesheet',
        href='https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&family=Lato&family=Roboto+Mono&display=swap'
    ),
    
    html.H1("EV Simulation Dashboard", style={'font-family': 'Roboto, sans-serif', 'font-size': '28px'}),
    
    html.Div([
        # Input Controls Column
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
                    min=1, max=1000, value=100, step=1,
                    marks={i: str(i) for i in range(0, 1001, 250)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'margin-top': '20px'}),
            html.Div([
                html.Label("Charging Stations: ", id='stations-label', style={'font-family': 'Lato, sans-serif', 'font-size': '16px', 'padding': '10px 0px'}),
                dcc.Slider(
                    id='stations-slider',
                    min=1, max=100, value=10, step=1,
                    marks={i: str(i) for i in range(0, 101, 25)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'margin-top': '20px'}),
            html.Div([
                html.Label("Chargers per Station: ", id='chargers-label', style={'font-family': 'Lato, sans-serif', 'font-size': '16px', 'padding': '10px 0px'}),
                dcc.Slider(
                    id='chargers-slider',
                    min=1, max=12, value=3, step=1,
                    marks={i: str(i) for i in range(1, 13)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'margin-top': '20px'}),
            html.Button('Run Simulation', id='run-button', n_clicks=0, 
                        style={'margin-top': '20px', 'font-family': 'Roboto, sans-serif'})
        ], style={'width': '20%', 'float': 'left', **COLUMN_STYLE}),
        
        # Results Column
        html.Div([
            html.H2("Results", style={'font-family': 'Roboto, sans-serif', 'font-size': '20px'}),
            html.Div([
                html.H3("Dead Batteries", style={'font-family': 'Roboto, sans-serif', 'font-size': '18px'}),
                html.Div(id='dead-batteries', style={'font-family': 'Roboto Mono, monospace', 'font-size': '24px'}),
            ]),
            html.Div([
                html.H3("Trip Time", style={'font-family': 'Roboto, sans-serif', 'font-size': '18px'}),
                html.Div(id='avg-trip-time', style={'font-family': 'Roboto Mono, monospace', 'font-size': '24px'}),
                dcc.Graph(id='trip-time-hist', style={'height': '30vh'})
            ]),
            html.Div([
                html.H3("Charging Time", style={'font-family': 'Roboto, sans-serif', 'font-size': '18px'}),
                html.Div(id='avg-charging-time', style={'font-family': 'Roboto Mono, monospace', 'font-size': '24px'}),
                dcc.Graph(id='charging-time-hist', style={'height': '30vh'})
            ]),
            html.Div([
                html.H3("Mileage", style={'font-family': 'Roboto, sans-serif', 'font-size': '18px'}),
                html.Div(id='avg-mileage', style={'font-family': 'Roboto Mono, monospace', 'font-size': '24px'}),
                dcc.Graph(id='mileage-hist', style={'height': '30vh'})
            ])
        ], style={'width': '35%', 'float': 'left', **COLUMN_STYLE}),
        
        # Map Column
        html.Div([
            html.H2("Route Map", style={'font-family': 'Roboto, sans-serif', 'font-size': '20px'}),
            dcc.Graph(id='route-map', style={'height': '100vh'})
        ], style={'width': '35%', 'float': 'right', **COLUMN_STYLE}),
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
     Output('avg-charging-time', 'children'),
     Output('avg-mileage', 'children'),
     Output('dead-batteries', 'children'),
     Output('trip-time-hist', 'figure'),
     Output('charging-time-hist', 'figure'),
     Output('mileage-hist', 'figure'),
     Output('route-map', 'figure')],
    [Input('run-button', 'n_clicks')],
    [State('route-dropdown', 'value'),
     State('drivers-slider', 'value'),
     State('stations-slider', 'value'),
     State('chargers-slider', 'value')]
)

def update_output(n_clicks, route, num_drivers, num_stations, num_chargers): # in the order of the State inputs
    if n_clicks > 0:

        # Create a SimulationParameters object with the input values
        params = SimulationParameters()
        params.num_drivers = num_drivers
        params.num_stations = num_stations
        params.num_chargers = num_chargers
        
        # Set the route distance based on the selected route
        if route == 'nyc_boston':
            params.loc_destination = 348  # km
        elif route == 'la_sf':
            params.loc_destination = 615  # km 
        elif route == 'chicago_detroit':
            params.loc_destination = 455  # km 
        
        # Run the simulation
        results = run_simulation(params)

        trip_time_hist = px.histogram(x=results['trip_times'], nbins=20)
        charging_time_hist = px.histogram(x=results['charging_times'], nbins=20)
        mileage_hist = px.histogram(x=results['mileages'], nbins=20)
        
        # Dummy map for now
        map_fig = go.Figure(go.Scattermapbox(
            mode = "markers+lines",
            lon = [-74.0060, -71.0589],
            lat = [40.7128, 42.3601],
            marker = {'size': 10}))

        map_fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=dict(lon=-72.5, lat=41.5),
                zoom=6
            )
        )
        
        return (
            f"Avg: {results['avg_trip_time']:.1f} min",
            f"Avg: {results['avg_charging_time']:.1f} min",
            f"Avg: {results['avg_mileage']:.1f} km",
            f"Count: {results['dead_batteries']}",
            trip_time_hist,
            charging_time_hist,
            mileage_hist,
            map_fig
        )
    
    # Return empty figures if the simulation hasn't been run
    return "N/A", "N/A", "N/A", "N/A", {}, {}, {}, {}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
