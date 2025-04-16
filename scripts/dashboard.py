# dashboard.py
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Load data
df = pd.read_csv('data/processed_data/falcon9_processed.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("SpaceX Launch Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.H3("Launch Site:"),
            dcc.Dropdown(
                id='site-dropdown',
                options=[
                    {'label': 'All Sites', 'value': 'ALL'},
                    {'label': 'CCAFS SLC-40', 'value': 'CCAFS SLC-40'},
                    {'label': 'KSC LC-39A', 'value': 'KSC LC-39A'},
                    {'label': 'VAFB SLC-4E', 'value': 'VAFB SLC-4E'}
                ],
                value='ALL',
                placeholder="Select a Launch Site"
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("Payload Range (kg):"),
            dcc.RangeSlider(
                id='payload-slider',
                min=0,
                max=df['payload_mass_kg'].max(),
                step=1000,
                value=[0, df['payload_mass_kg'].max()],
                marks={i: f'{i}kg' for i in range(0, int(df['payload_mass_kg'].max()) + 1, 5000)}
            )
        ], style={'width': '48%', 'display': 'inline-block'})
    ]),
    
    html.Div([
        html.Div([
            dcc.Graph(id='success-pie-chart')
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='success-payload-scatter-chart')
        ], style={'width': '48%', 'display': 'inline-block'})
    ]),
    
    html.Div([
        dcc.Graph(id='yearly-trend-chart')
    ])
])

# Callback for pie chart
@app.callback(
    Output(component_id='success-pie-chart', component_property='figure'),
    [Input(component_id='site-dropdown', component_property='value')]
)
def update_pie_chart(selected_site):
    if selected_site == 'ALL':
        fig = px.pie(df, names='landing_success', 
                     title='Total Success vs Failure for All Sites',
                     color_discrete_map={0: 'red', 1: 'green'},
                     labels={0: 'Failure', 1: 'Success'})
    else:
        filtered_df = df[df['launch_site'] == selected_site]
        fig = px.pie(filtered_df, names='landing_success', 
                     title=f'Success vs Failure for {selected_site}',
                     color_discrete_map={0: 'red', 1: 'green'},
                     labels={0: 'Failure', 1: 'Success'})
    return fig

# Callback for scatter chart
@app.callback(
    Output(component_id='success-payload-scatter-chart', component_property='figure'),
    [Input(component_id='site-dropdown', component_property='value'),
     Input(component_id='payload-slider', component_property='value')]
)
def update_scatter_chart(selected_site, payload_range):
    low, high = payload_range
    mask = (df['payload_mass_kg'] >= low) & (df['payload_mass_kg'] <= high)
    
    if selected_site == 'ALL':
        filtered_df = df[mask]
    else:
        filtered_df = df[mask & (df['launch_site'] == selected_site)]
    
    fig = px.scatter(filtered_df, 
                    x='payload_mass_kg', 
                    y='flight_number',
                    color='landing_success',
                    title='Correlation between Payload and Flight Number',
                    color_discrete_map={0: 'red', 1: 'green'})
    
    return fig

# Callback for yearly trend chart
@app.callback(
    Output(component_id='yearly-trend-chart', component_property='figure'),
    [Input(component_id='site-dropdown', component_property='value')]
)
def update_trend_chart(selected_site):
    if selected_site == 'ALL':
        yearly_data = df.groupby('launch_year')['landing_success'].agg(['mean', 'count']).reset_index()
    else:
        filtered_df = df[df['launch_site'] == selected_site]
        yearly_data = filtered_df.groupby('launch_year')['landing_success'].agg(['mean', 'count']).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yearly_data['launch_year'],
        y=yearly_data['mean'],
        mode='lines+markers',
        name='Success Rate',
        yaxis='y'
    ))
    
    fig.add_trace(go.Bar(
        x=yearly_data['launch_year'],
        y=yearly_data['count'],
        name='Number of Launches',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Yearly Launch Success Rate and Count',
        xaxis=dict(title='Year'),
        yaxis=dict(
            title='Success Rate',
            range=[0, 1]
        ),
        yaxis2=dict(
            title='Number of Launches',
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.1, y=1.1, orientation='h')
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)