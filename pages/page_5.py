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
import dash_ag_grid as dag
import math
import random
import ast
from algorithms import apriori, generate_association_rules

dash.register_page(__name__, path='/clustering',name='Clustering')

data = pd.read_csv('./new_dataset_1.csv')
columns = data.columns

####################################################### LAYOUT COMPONENT ####################################################################
algorithm_dropdown = html.Div([
    html.H6(" Choose an Algorithm and set its Parameters"),
    dcc.Dropdown(
    id='clust-algorithm-dropdown',
    options=[
        "KMEANS","DBSCAN"
    ],
    value="KMEANS",
    className="text-success",
)])

parameters_kmeans = html.Div([
    html.H6("select K and distance type"),
    dbc.Input(id="k", type="number", placeholder="k", min=1, max=10, step=1, value=3, style={'width': '90%'}),
    dcc.Dropdown(id="distance-type",options=[
        "euclidean","manhattan","cosine","minkowski"
    ],value="minkowski"),
])
parameters_dbscan = html.Div([
    html.H6("select eps and minpts"),
    dbc.Input(id="eps", type="number", placeholder="eps", min=0.5, max=10, step=0.5, value=0.5, style={'width': '90%'}),
    dbc.Input(id="minpts", type="number", placeholder="minpts", min=1, max=10, step=1, value=5, style={'width': '90%'}),
])

def get_parameters(algorithm="KMEANS"):
    if algorithm == "KMEANS":
        return parameters_kmeans
    elif algorithm == "DBSCAN":
        return parameters_dbscan

parameters = dbc.Card([dbc.CardBody(algorithm_dropdown),
                        dbc.CardBody(html.Div([get_parameters()])
                        ,id="clust-parameters-body")],color="light", outline=True)
sample_input= html.Div([html.H6("Test on a new Sample"),
                         html.Div([dbc.Input(type="number",id=f"clust-sample-input-{i}",placeholder=f"{i}",min=0,className="me-1") for i in columns[1:]],
                                                                  className="d-flex justify-content-between")],id="clust-sample-input")
sample_input_card = dbc.Card([dbc.CardBody(sample_input)],color="light", outline=True)

####################################################### LAYOUT ####################################################################

layout = dbc.Container([
    html.H1("Clustering"),
    html.Hr(),
    dbc.Row([
        dbc.Col(parameters,md=4),
    ]),
],fluid=True)

############################## CALLBACK ########################   
@callback(
    Output("clust-parameters-body", "children"),
    Input("clust-algorithm-dropdown", "value")
)
def update_parameters(algorithm):
    return get_parameters(algorithm)
