import dash
from dash import Dash, html, dash_table, dcc, Input, Output, State, callback,callback_context
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
import joblib
from algorithms import K_means as KMEANS
from algorithms import DBSCAN, split_train_test, data_to_data_2d
from metrics import calculate_silhouette,inter_cluster,intra_cluster

dash.register_page(__name__, path='/clustering',name='Clustering')

data = pd.read_csv('./new_dataset.csv')
columns = data.columns
observation = [264,10.3,475,7.49,0.74,10.56,0.45,7.36,1.87,10.63,0.63,1.5136]
data_2d = data_to_data_2d(data)

######################################################## Functions ####################################################################
def clust_instanciate_model(data,algorithme_type,k,nb_iterations,distance_type,epsilon,minpts):
    if algorithme_type=="KMEANS":
            if k==3 and nb_iterations==50 and distance_type=="minkowski":
                return './models/kmeans_model.pkl'
            model= KMEANS(k,nb_iterations,distance_type)
            model.fit(data)
            joblib.dump(model, './models/kmeans_model_new.pkl')
            return './models/kmeans_model_new.pkl'
    elif algorithme_type=="DBSCAN":
            if epsilon==0.7 and minpts==15:
                return './models/dbscan_cluster.pkl'
            model= DBSCAN(epsilon,minpts)
            model.fit(data)
            joblib.dump(model, './models/dbscan_cluster_new.pkl')
            return './models/dbscan_cluster_new.pkl'
            
def clust_train_predict(data=data,algorithme_type="KMEANS",k=3,nb_iterations=50,distance_type="minkowski",epsilon=0.7,minpts=15):
    if algorithme_type=="KMEANS":
        model_trained= clust_instanciate_model(data,algorithme_type,k,nb_iterations,distance_type,epsilon,minpts)
    elif algorithme_type=="DBSCAN":
        model_trained=clust_instanciate_model(data,algorithme_type,k,nb_iterations,distance_type,epsilon,minpts)
    model = joblib.load(model_trained)
    labels = model.labels_
    data_clust = model.data
    _,silhouette = calculate_silhouette(data_clust)
    intra_c = intra_cluster(data, labels)
    inter_c = inter_cluster(data, labels)
    results = [f"Silhouette: {silhouette}", f"Intra Cluster: {intra_c}", f"Inter Cluster: {inter_c}"]
    return model_trained,labels,html.Div(html.Ul([html.Li(f"{results[i]}") for i in range(len(results))]),)
    
    

def clustering_results_plot(labels,data_2d = data_2d):
    features_to_visualize = data_2d.iloc[:, :2].values
    scatter = go.Scatter(
    x=features_to_visualize[:, 0],
    y=features_to_visualize[:, 1],
    mode='markers',
    marker=dict(
        color=labels,
        colorscale='Viridis',
        opacity=0.5
    ),
    text=data_2d.index  
    )

    layout = go.Layout(
        title='K-means Clustering Results with k clusters',
        xaxis=dict(title=''),
        yaxis=dict(title=''),  
        showlegend=False,
    )

    # Create a figure and add the scatter trace
    fig = go.Figure(data=[scatter], layout=layout)
    return fig


####################################################### LAYOUT COMPONENT ####################################################################
parameters_kmeans = [
    html.H6("select K and distance type"),
    dbc.Input(id="k", type="number", placeholder="k", min=1, step=1, value=3,className="mb-1"),
    dcc.Dropdown(id="kmeans-distance-type",options=[
        "euclidean","manhattan","cosine","minkowski"
    ],value="minkowski",className="mb-1"),
    dbc.Input(id="nb-iteration", type="number", placeholder="nb-iteration", min=1, step=1, value=50, className="mb-1"),
    dbc.Button("Train", id="kmeans-train-button", n_clicks=0, className="mb-1"),
]
kmeans_ct = dbc.Card([dbc.CardBody(parameters_kmeans),])

parameters_dbscan = [
    html.H6("select eps and minpts"),
    dbc.Input(id="eps", type="number", placeholder="epsilon", min=0,value=0.7,className="mb-1"),
    dbc.Input(id="minpts", type="number", placeholder="minPointss", min=0, step=1, value=15, className="mb-1"),
    dbc.Button("Train", id="dbscan-train-button", n_clicks=0, className="mb-1"),
]
dbscan_ct = dbc.Card([dbc.CardBody(parameters_dbscan),])

parameter_tabs = dbc.Tabs(
    [
        dbc.Tab(kmeans_ct, label="KMEANS"),
        dbc.Tab(dbscan_ct, label="DBSCAN"),
    ]
)

sample_input= html.Div([html.H6("Test on a new Sample"),
                         html.Div([dbc.Input(type="number",id=f"clust-sample-input-{i}",placeholder=f"{i}",min=0,className="me-1") for i in columns[1:]],
                                                                  className="d-flex justify-content-between")],id="clust-sample-input")
sample_input_card = dbc.Card([dbc.CardBody(sample_input)],color="light", outline=True)

####################################################### LAYOUT ####################################################################

layout = dbc.Container([
    html.H1("Clustering"),
    html.Hr(),
    dcc.Store(id="clust-current-model",data="",storage_type="session"),
    dbc.Row([
        dbc.Col(dbc.Card([html.H6("Choose an Algorithm and set its Parameters",className="pt-2 text-center"),parameter_tabs]),width=4),
        dbc.Col(id="clust-metrics-output"),
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id="clust-plot"),width=10)]),
],fluid=True, className="dbc dbc-ag-grid")

############################## CALLBACK ########################   



@callback(
    [Output("clust-current-model", "data"),Output("clust-metrics-output", "children"),Output("clust-plot", "figure")],
    [Input("kmeans-train-button", "n_clicks"),Input("dbscan-train-button", "n_clicks")],
    [State("k", "value"),State("kmeans-distance-type", "value"),State("nb-iteration", "value"),
     State("eps", "value"),State("minpts", "value")],
)
def update_clust_instnance_model(bt1,bt2,k,distance_type,nb_iteration,eps,minpts):
    if callback_context.triggered_id == "kmeans-train-button":
        model,labels,metrics = clust_train_predict(data=data,algorithme_type="KMEANS",k=k,distance_type=distance_type,nb_iterations=nb_iteration)
        fig_1 = clustering_results_plot(labels)
        return model, metrics, fig_1
    elif callback_context.triggered_id == "dbscan-train-button":
        model,labels,metrics = clust_train_predict(data=data,algorithme_type="DBSCAN",epsilon=eps,minpts=minpts)
        fig_1 = clustering_results_plot(labels)
        return model, metrics, fig_1
    else:
        model = "./models/kmeans_model.pkl"
        _,labels,metrics = clust_train_predict(data=data,algorithme_type="KMEANS",k=k,distance_type=distance_type,nb_iterations=nb_iteration)
        fig_1 = clustering_results_plot(labels)
        
        return model, metrics, fig_1

