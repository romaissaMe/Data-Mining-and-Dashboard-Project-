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
import joblib
from algorithms import KNN, DecisionTree, RandomForest, split_train_test
from metrics import accuracy, precision, recall,f_score,specificity,calculate_silhouette,confusion_matrix

dash.register_page(__name__, path='/ML',name='Classification')
data = pd.read_csv('./new_dataset_1.csv')
data_discretise = pd.read_csv('./new_dataset_discretise.csv')
columns = data.columns
X_train_1, X_test_1, y_train_1, y_test_1 = split_train_test(data_discretise)
X_train_2, X_test_2, y_train_2, y_test_2 = split_train_test(data)
classes = np.unique(y_train_2)
###################################################### Functions #######################################################
def instanciate_model(algorithme_type,k,distance_type,t_min_samples_split,t_max_depth,min_samples_split,max_depth,nb_trees):
    if algorithme_type=="KNN":
        if k==30 and distance_type =="euclidean":
            return 'knn_model.pkl'
        else:
            model= KNN(k,distance_type)
            model= model.fit(X_train_2, y_train_2)
            joblib.dump(model, 'knn_model_new.pkl')
            return 'knn_model_new.pkl'
    elif algorithme_type=="DECISON TREE":
        if t_max_depth==50 and t_min_samples_split==30:
            return 'decision_tree_model.pkl'
        else:
            model=DecisionTree(t_min_samples_split,t_max_depth)
            model = model.fit(X_train_1, y_train_1)
            joblib.dump(model, 'decision_tree_model_new.pkl')
            return './decision_tree_model_new.pkl'
    elif algorithme_type=="RANDOM FORESTS":
        if max_depth==150 and min_samples_split==35 and nb_trees==10:
            return 'random_forest_model.pkl'
        else:
            model= RandomForest(min_samples_split=min_samples_split,max_depth=max_depth)
            model=model.fit(X_train_1, y_train_1)
            joblib.dump(model, 'random_forest_model_new.pkl')
            return 'random_forest_model_new.pkl'

def calc_accuracy(y_test, predictions):
    return accuracy(y_test, predictions)

def calc_precision(y_test, predictions):
    return precision(y_test, predictions)

def calc_recall(y_test, predictions):
    return recall(y_test, predictions)

def calc_f_score(y_test, predictions):
    return f_score(y_test, predictions)

def calc_confusion_matrix(y_test, predictions):
    return confusion_matrix(y_test, predictions)

def calc_specificity(y_test, predictions):
    return specificity(y_test, predictions)

def calc_silhouette(y_test, predictions):
    return calculate_silhouette(y_test, predictions)

def train_predict(algorithme_type="DECISON TREE",k=30,distance_type="euclidean",t_min_samples_split=30,t_max_depth=50,min_samples_split=35,max_depth=150,nb_trees=10):
    model_trained = instanciate_model(algorithme_type,k,distance_type,t_min_samples_split,t_max_depth,min_samples_split,max_depth,nb_trees)
    model = joblib.load(model_trained)
    if algorithme_type=="KNN":
        predictions = model.predict(X_test_2)
        accuracy = calc_accuracy(y_test_2, predictions)
        precision = calc_precision(y_test_2, predictions)
        recall = calc_recall(y_test_2, predictions)
        f_score = calc_f_score(y_test_2, predictions)
        confusion_matrix= calc_confusion_matrix(y_test_2, predictions)
    elif algorithme_type=="DECISON TREE":
        predictions = model.predict(X_test_1)
        accuracy = calc_accuracy(y_test_1, predictions)
        precision = calc_precision(y_test_1, predictions)
        recall = calc_recall(y_test_1, predictions)
        f_score = calc_f_score(y_test_1, predictions)
        confusion_matrix= calc_confusion_matrix(y_test_1, predictions)
    elif algorithme_type=="RANDOM FORESTS":
        predictions = model.predict(X_test_1)
        accuracy = calc_accuracy(y_test_1, predictions)
        precision = calc_precision(y_test_1, predictions)
        recall = calc_recall(y_test_1, predictions)
        f_score = calc_f_score(y_test_1, predictions)
        confusion_matrix= calc_confusion_matrix(y_test_1, predictions)
    return [accuracy,precision,recall,f_score,confusion_matrix]

def make_prediction(model='./decision_tree_model.pkl',observation=[264,10.3,475,7.49,0.74,10.56,0.45,7.36,1.87,10.63,0.63,1.5136]):
    model = joblib.load(model)
    return model.predict([observation])



####################################################### LAYOUT COMPONENT ####################################################################
algorithm_dropdown = html.Div([
    html.H6(" Choose an Algorithm and set its Parameters"),
    dcc.Dropdown(
    id='algorithm-dropdown',
    options=
       [ {"label": i,"value":i} for i in ["KNN","DECISON TREE","RANDOM FORESTS"]],
    value="DECISON TREE",
)])
parameters_knn = html.Div([
    html.H6("select K, distance type and number of iteration"),
    dbc.Input(id="knn-k", type="number", placeholder="k", min=1, step=1, value=2,name="k"),
    dcc.Dropdown(id="distance-type",options=[{"label": i,"value":i} for i in ["euclidean","manhattan","cosine","minkowski"]],
                 value="euclidean"),
])

parameters_dt = html.Div([
    html.H6("select max depth, min sample split"),
    dbc.Input(id="dt-max-depth", type="number", placeholder="max_depth", min=1,  step=1, value=50, style={'width': '90%'}),
    dbc.Input(id="dt-min-samples-split", type="number", placeholder="min_samples_split", min=1, step=1, value=30, style={'width': '90%'}),
])
parameters_rf = html.Div([
    html.H6("select number of trees, max depth and min sample split"),
    dbc.Input(id="n_trees", type="number", placeholder="n_trees", min=1, step=1, value=10, style={'width': '90%'}),
    dbc.Input(id="rf-max-depth", type="number", placeholder="max_depth", min=1, step=1, value=150, style={'width': '90%'}),
    dbc.Input(id="rf-min-samples-split", type="number", placeholder="min_samples_split", min=1,  step=1, value=35, style={'width': '90%'}),
])
def get_parameters(algorithm="KNN"):
    if algorithm == "KNN":
        return parameters_knn
    elif algorithm == "DECISON TREE":
        return parameters_dt
    elif algorithm == "RANDOM FORESTS":
        return parameters_rf

parameters = dbc.Card([dbc.CardBody(algorithm_dropdown),
                        dbc.CardBody(html.Div([get_parameters()]),id="parameters-body"),
                        dbc.CardBody(dbc.Button('Train Model', id='train-button'))
                        ],color="light", outline=True)
sample_input= html.Div([html.H6("Test on a new Sample"),
                         html.Div([dbc.Input(type="number",id=f"sample-input-{i}",placeholder=f"{i}",min=0,className="me-1") for i in columns[1:]],
                                                                  className="d-flex justify-content-between")],id="sample-input")
sample_input_card = dbc.Card([dbc.CardBody(sample_input),dbc.CardBody([dbc.Button('Predict', id='predict-button')])],id="sample-input-card",color="light", outline=True)

####################################################### LAYOUT ####################################################################

layout = dbc.Container([
    html.H1("Classification"),
    html.Hr(),
    dcc.Store(id="current-model",data="",storage_type="session"),
    dbc.Row([
        dbc.Col(parameters,md=4),
    ]),
    dbc.Row([html.Div([i for i in train_predict()],id="metrics-output")]),
     dbc.Row([dbc.Col([sample_input_card]),dbc.Col([make_prediction()],id="prediction-output")]),
],fluid=True)

####################################################### CALLBACK ####################################################################
@callback(
    Output("parameters-body", "children"),
    Input("algorithm-dropdown", "value")
)
def update_parameters(algorithm):
    if algorithm == "KNN":
        return parameters_knn
    elif algorithm == "DECISON TREE":
        return parameters_dt
    elif algorithm == "RANDOM FORESTS":
        return parameters_rf
    
@callback(
    Output("current-model", "data"),
    Input("train-button", "n_clicks"),
    [State("algorithm-dropdown", "value"),State("knn-k", "value"),State("distance-type", "value"),
     State("dt-min-samples-split", "value"),State("dt-max-depth", "value"),
     State("rf-max-depth", "value"),State("rf-min-samples-split", "value"),State("n_trees", "value")]
)
def update_train_model(algorithm,n_clicks,knn_k,distance_type,dt_min_samples_split,dt_max_depth,rf_min_samples_split,rf_max_depth,n_trees):
    if n_clicks is None:
        return None
    model = train_predict(algorithm,knn_k,distance_type,dt_min_samples_split,dt_max_depth,rf_min_samples_split,rf_max_depth,n_trees)
    return model

@callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    [State("current-model", "data"),State("sample-input-N", "value"),
     State("sample-input-P", "value"),State("sample-input-K", "value"),State("sample-input-EC", "value"),
     State("sample-input-pH", "value"), State("sample-input-S", "value"),State("sample-input-Zn", "value"),
     State("sample-input-Fe", "value"),State("sample-input-Cu", "value"),State("sample-input-Mn", "value"),
     State("sample-input-B", "value"),State("sample-input-OM", "value")])
def update_make_predcion(model_name,n_clicks,N,P,K,EC,pH,S,Zn,Fe,Cu,Mn,B,OM):
    if n_clicks is None:
        return None
    observation = [N,P,K,EC,pH,S,Zn,Fe,Cu,Mn,B,OM]
    prediction = make_prediction(model_name,observation)
    
   