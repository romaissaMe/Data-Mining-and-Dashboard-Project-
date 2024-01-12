import dash
from dash import Dash, html, dash_table, dcc, Input, Output, State, callback
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dash import dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/covid_database',name='Covid-19 Database')

data= pd.read_csv("./new_dataset_2.csv")
data['Start date']= pd.to_datetime(data['Start date'])
data['end date']= pd.to_datetime(data['end date'])
attribute_dropdown = dcc.Dropdown(id='attribute-dropdown', className='attributes-dropdown',options=list(["case count","test count","positive tests","case rate","test rate","positivity rate"]),value='case count',style={'width':'150px'})
zone_dropdown = dcc.Dropdown(id='zone-dropdown', className='attributes-dropdown mb-1',options=list(data['zcta'].unique()),value=94087)
season_dropdown = dcc.Dropdown(id='season-dropdown',className='attributes-dropdown  mb-1',options=list(['monthly','weekly','yearly']),value='monthly')
period_dropdown = dcc.Dropdown(id='period-dropdown', className='attributes-dropdown  mb-1',options=list(data['time_period'].unique()),value=26)
################################################## UTILITY FUNCTIONS #####################################################################
def global_line_chart(data=data,attribute='case count'):
    grouped_data = data.groupby('time_period')[attribute].sum().reset_index()
    fig = px.line(grouped_data, x='time_period', y=f'{attribute}', 
                labels={f'{attribute}': attribute, 'time_period': 'Time Period'})
    fig.update_layout(
        xaxis_title='Time Period',
        yaxis_title=f'{attribute}',
        margin=dict(l=0, r=0, t=0, b=1),  
        font=dict(size=10),
    )
    return fig

def line_chart_per_zone(data=data,attribute='case count',zone=94087):
    data = data.loc[data['zcta']==zone]
    grouped_data = data.groupby('time_period')[attribute].sum().reset_index()
    fig = px.line(grouped_data, x='time_period', y=f'{attribute}', 
                labels={f'{attribute}': attribute, 'time_period': 'Time Period'})
    fig.update_layout(
        xaxis_title='Time Period',
        yaxis_title=f'{attribute}',
        margin=dict(l=0, r=0, t=0, b=0),  
        font=dict(size=10),  
    )
    return fig

def monthly_traces_per_zone(data,zone):
    data = data[data['zcta'] == zone]
    sorted_data = data.sort_values(by='Start date')
    monthly_data = sorted_data.resample('M', on='Start date')[['case count', 'positive tests']].sum()

    # Create traces for Case Count and Positive Tests
    trace_case_count = go.Scatter(x=monthly_data.index, y=monthly_data['case count'], mode='lines', name='Case Count')
    trace_positive_tests = go.Scatter(x=monthly_data.index, y=monthly_data['positive tests'], mode='lines', name='Positive Tests')

    # Layout settings
    layout = go.Layout(
        title=f'Monthly Case Count and Positive Tests for Zone: {zone}',
        xaxis=dict(title='Months'),
        yaxis=dict(title='Counts'),
        legend=dict(x=0, y=1) 
    )
    fig = go.Figure(data=[trace_case_count, trace_positive_tests], layout=layout)
    return fig

def weekly_traces_per_zone(data,zone):
    data = data[data['zcta'] == zone]
    sorted_data = data.sort_values(by='Start date')
    monthly_data = sorted_data.resample('W-Mon', on='Start date')[['case count', 'positive tests']].sum()

    # Create traces for Case Count and Positive Tests
    trace_case_count = go.Scatter(x=monthly_data.index, y=monthly_data['case count'], mode='lines', name='Case Count')
    trace_positive_tests = go.Scatter(x=monthly_data.index, y=monthly_data['positive tests'], mode='lines', name='Positive Tests')

    # Layout settings
    layout = go.Layout(
        title=f'weekly Case Count and Positive Tests for Zone: {zone}',
        xaxis=dict(title='Months'),
        yaxis=dict(title='Counts'),
        legend=dict(x=0, y=1) 
    )
    fig = go.Figure(data=[trace_case_count, trace_positive_tests], layout=layout)
    return fig
def yearly_traces_per_zone(data,zone):
    data = data[data['zcta'] == zone]
    sorted_data = data.sort_values(by='Start date')
    monthly_data = sorted_data.resample('Y', on='Start date')[['case count', 'positive tests']].sum()

    # Create traces for Case Count and Positive Tests
    trace_case_count = go.Scatter(x=monthly_data.index, y=monthly_data['case count'], mode='lines', name='Case Count')
    trace_positive_tests = go.Scatter(x=monthly_data.index, y=monthly_data['positive tests'], mode='lines', name='Positive Tests')

    # Layout settings
    layout = go.Layout(
        title=f'Yearly Case Count and Positive Tests for Zone: {zone}',
        xaxis=dict(title='Years'),
        yaxis=dict(title='Counts'),
        legend=dict(x=0, y=1) 
    )
    fig = go.Figure(data=[trace_case_count, trace_positive_tests], layout=layout)
    return fig

def seasonal_traces_per_zone(data=data,zone=94087,period='monthly'):
    if period == 'weekly':
        fig = weekly_traces_per_zone(data,zone)
        return fig
    elif period == 'monthly':
        fig = monthly_traces_per_zone(data,zone)
        return fig
    elif period == 'yearly':
        fig = yearly_traces_per_zone(data,zone)
        return fig

def top_zones_bar_plot(data=data):
    top_5_zones = data.groupby('zcta')['case count'].sum().nlargest(5).reset_index()
    top_5_zones['zcta']=top_5_zones['zcta'].astype('str')
    fig = px.bar(top_5_zones, x='zcta', y='case count', 
                labels={'zcta': 'Zone', 'case count': 'Total Cases'},
                title='Top 5 Zones Most Affected by COVID-19')
    return fig

def heatmap_zone(data=data):
    grouped_data = data.groupby('zcta')[['case count', 'positive tests']].sum().reset_index()
    grouped_data['zcta']=grouped_data['zcta'].astype('str')
    fig = px.imshow(grouped_data[['case count', 'positive tests']].T,
                    labels=dict(x="zone", y="Attribut", color="nombre total"),
                    x=grouped_data['zcta'],
                    y=['case count', 'positive tests'],
                    color_continuous_scale='Viridis',
                    title='Total number distribution of confirmed cases and positive tests per zone')

    fig.update_xaxes(title_text='Zone')
    fig.update_yaxes(title_text='Attribut')
    return fig

def distribution_year_per_zone(data=data):
    data_with_year = data.copy()
    data_with_year['year']=data_with_year['Start date'].dt.year
    data_with_year['year']= data_with_year['year'].astype('str')
    grouped_data = data_with_year.groupby(['zcta','year'])['case count'].sum().reset_index()
    grouped_data['zcta']=grouped_data['zcta'].astype('str')
    fig = px.bar(grouped_data, x='zcta', y='case count', color='year')
    fig.update_layout(title='Distribution cas Covid positifs par zone et par année')
    return fig

def population_test_heatmap(data=data):
    fig = px.density_heatmap(data, x='population', y='test count', title='Population vs Test Count Heatmap')
    return fig

def case_test_pos_test_relationship_per_period(data=data,period=26):
    dataset = data.loc[data['time_period'] == period,data.columns]
    dataset['zcta']=dataset.loc[:,'zcta'].astype('str')
    fig = px.bar(dataset, x='zcta', y=['case count', 'test count', 'positive tests'], 
                labels={'value': 'Count'},barmode='group', 
                title='Relationship between Confirmed Cases, Tests Conducted, and Positive Tests Over Time')
    return fig

################################################################ Page Layout ################################################################

layout= dbc.Container([
    dbc.Row([dbc.Col([attribute_dropdown]),dbc.Col([zone_dropdown])],className='mb-3'),
    dbc.Row([
        dbc.Col([dcc.Graph(id='global-line-chart',figure=global_line_chart(),
                           style={'height': '100%', 'width': '100%'})],width=5),
        dbc.Col([dcc.Graph(id='line-chart-per-zone',figure=line_chart_per_zone())]),   
                           
            ],className='h-25 mb-1'),

    dbc.Row([
        season_dropdown,
        dbc.Col([dcc.Graph(figure=seasonal_traces_per_zone(),id='seasonal-traces-per-zone')]),
    ],className='mb-1'),
    dbc.Row([
        dbc.Col([dcc.Graph(figure=heatmap_zone(),id='heatmap-zone')]),
        dbc.Col([dcc.Graph(figure=distribution_year_per_zone(),id='distribution-year-per-zone')]),
        dbc.Col([dcc.Graph(id='top-zones-bar-plot',figure=top_zones_bar_plot())]),
    ],className='mb-1'),
    dbc.Row([
        dbc.Col([dcc.Graph(figure=population_test_heatmap(),id='population-test-heatmap')],width=4),
        dbc.Col([period_dropdown,dcc.Graph(figure=case_test_pos_test_relationship_per_period(),id='case-test-pos-test-relationship-per-period',className='mt-1'),])
       
    ])
        
],fluid=True,
    className="dbc dbc-ag-grid")


##################################################### Callbacks ##################################################################################################

@callback(
    Output('global-line-chart', 'figure'),
    [Input('attribute-dropdown', 'value')],
)
def update_global_line_chart(attribute):
    fig = global_line_chart(data,attribute)
    return fig

@callback(
    Output('line-chart-per-zone', 'figure'),
    [Input('attribute-dropdown', 'value'),Input('zone-dropdown', 'value')],
)
def update_line_chart_per_zone(attribute,zone):
    fig = line_chart_per_zone(data,attribute,zone)
    return fig

@callback(
    Output('seasonal-traces-per-zone','figure'),
    [Input('season-dropdown','value'),Input('zone-dropdown', 'value')]
)
def update_seasonal_traces_per_zone(selected_season,selected_zone):
    fig= seasonal_traces_per_zone(data,selected_zone,selected_season)
    return fig

@callback(
    Output('case-test-pos-test-relationship-per-period','figure'),
    [Input('period-dropdown','value')]
)
def update_period_traces_per_zone(selected_period):
    fig= case_test_pos_test_relationship_per_period(data,selected_period)
    return fig
