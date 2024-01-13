import dash
from dash_extensions.enrich import Dash, html, dash_table, dcc, Input, Output, State
from dash import callback, callback_context
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from dash import dcc
import dash_bootstrap_components as dbc
import joblib
from algorithms import KNN, DecisionTree, RandomForest, split_train_test
from metrics import accuracy, precision, recall,f_score,specificity,calculate_silhouette,confusion_matrix

dash.register_page(__name__, path='/classification',name='Classification')
data = pd.read_csv('./new_dataset_1.csv')
data_discretise = pd.read_csv('./new_dataset_discretise.csv')
columns = data.columns
X_train_1, X_test_1, y_train_1, y_test_1 = split_train_test(data_discretise)
X_train_2, X_test_2, y_train_2, y_test_2 = split_train_test(data)
classes = np.unique(y_train_2)
observation = [264,10.3,475,7.49,0.74,10.56,0.45,7.36,1.87,10.63,0.63,1.5136]
###################################################### Functions #######################################################
def instanciate_model(algorithme_type,k,distance_type,t_min_samples_split,t_max_depth,min_samples_split,max_depth,nb_trees):
    if algorithme_type=="KNN":
        if k==30 and distance_type =="euclidean":
            return 'knn_model.pkl'
        else:
            model= KNN(k,distance_type)
            model.fit(X_train_2, y_train_2)
            joblib.dump(model, 'knn_model_new.pkl')
            return 'knn_model_new.pkl'
    elif algorithme_type=="DECISON TREE":
        if t_max_depth==50 and t_min_samples_split==30:
            return 'decision_tree_model.pkl'
        else:
            model=DecisionTree(t_min_samples_split,t_max_depth)
            model.fit(X_train_1, y_train_1)
            joblib.dump(model, 'decision_tree_model_new.pkl')
            return 'decision_tree_model_new.pkl'
    elif algorithme_type=="RANDOM FORESTS":
        if max_depth==150 and min_samples_split==35 and nb_trees==10:
            return 'random_forest_model.pkl'
        else:
            model= RandomForest(min_samples_split=min_samples_split,max_depth=max_depth)
            model.fit(X_train_1, y_train_1)
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
    elif algorithme_type=="DECISON TREE":
        predictions = model.predict(X_test_1)
        accuracy = calc_accuracy(y_test_1, predictions)
        precision = calc_precision(y_test_1, predictions)
        recall = calc_recall(y_test_1, predictions)
        f_score = calc_f_score(y_test_1, predictions)
    elif algorithme_type=="RANDOM FORESTS":
        predictions = model.predict(X_test_1)
        accuracy = calc_accuracy(y_test_1, predictions)
        precision = calc_precision(y_test_1, predictions)
        recall = calc_recall(y_test_1, predictions)
        f_score = calc_f_score(y_test_1, predictions)
    results = [f"Accuracy: {accuracy:.2f}", f"Precision: {precision:.2f}", f"Recall: {recall:.2f}", f"F1 Score: {f_score:.2f}"]
    return model_trained,predictions,html.Div(
        [
            html.Ul([html.Li(f"{results[i]}") for i in range(len(results))]),
        ], className=""
    )

def make_prediction(model='decision_tree_model.pkl',observation=[264,10.3,475,7.49,0.74,10.56,0.45,7.36,1.87,10.63,0.63,1.5136]):
    model = joblib.load(model)
    prediction = model.predict([observation])
    return prediction

def confusion_matrix_plot(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Create a list of strings for displaying values in annotations
    annotations = []
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            annotations.append(dict(x=j, y=i, text=str(conf_matrix[i][j]), showarrow=False))

    # Create a trace for the heatmap
    trace = go.Heatmap(z=conf_matrix,
                    x=['0', '1', '2'],  # Replace with your class labels
                    y=['0', '1', '2'],  # Replace with your class labels
                   )

    # Create a figure and add the trace with annotations
    layout = go.Layout(title='Confusion Matrix', xaxis=dict(title='Predicted'), yaxis=dict(title='Actual'), annotations=annotations)
    fig = go.Figure(data=[trace], layout=layout)
    return fig


def true_vs_predicted_labels_plot(y_test, y_pred):
    x_values = list(range(len(y_test)))  # Convert range to a list

    fig = go.Figure()

    # Add trace for true values (y_test)
    fig.add_trace(go.Scatter(x=x_values, y=y_test, mode='markers', name='True Values'))

    # Add trace for predicted values (y_pred)
    fig.add_trace(go.Scatter(x=x_values, y=y_pred, mode='markers', name='Predicted Values'))

    # Update marker colors for true and predicted values
    fig.update_traces(marker=dict(color='blue'), selector=dict(name='True Values'))
    fig.update_traces(marker=dict(color='red'), selector=dict(name='Predicted Values'))

    # Update layout
    fig.update_layout(
        title='True vs Predicted Values',
        xaxis_title='Index',
        yaxis_title='Values'
    )
    return fig


####################################################### LAYOUT COMPONENT ####################################################################
parameters_knn = [
    html.H6("select K, distance type and number of iteration"),
    dbc.Input(id="knn-k", type="number", placeholder="k", min=1, step=1, value=30,name="k",className="mb-1"),
    dcc.Dropdown(id="distance-type",options=[{"label": i,"value":i} for i in ["euclidean","manhattan","cosine","minkowski"]],
                 value="euclidean",className="text-dark"),
    dbc.Button('Train Model', id='knn-train-button',className="mt-1")
]
knn_content = dbc.Card([dbc.CardBody(parameters_knn),])

parameters_dt = [
    html.H6("select max depth, min sample split"),
    dbc.Input(id="dt-max-depth", type="number", placeholder="max_depth", min=1,  step=1, value=50,className="mb-1"),
    dbc.Input(id="dt-min-samples-split", type="number", placeholder="min_samples_split", min=1, step=1, value=30,className="mb-1"),
    dbc.Button('Train Model', id='dt-train-button')
]
dt_content = dbc.Card([dbc.CardBody(parameters_dt),])
parameters_rf = [
    html.H6("select number of trees, max depth and min sample split"),
    dbc.Input(id="n_trees", type="number", placeholder="n_trees", min=1, step=1, value=10,className="mb-1"),
    dbc.Input(id="rf-max-depth", type="number", placeholder="max_depth", min=1, step=1, value=150, className="mb-1"),
    dbc.Input(id="rf-min-samples-split", type="number", placeholder="min_samples_split", min=1,  step=1, value=35,className="mb-1"),
    dbc.Button('Train Model', id='rf-train-button')
]
rf_content = dbc.Card([dbc.CardBody(parameters_rf),])

sample_input= html.Div([html.H6("Test on a new Sample"),
                         html.Div([dbc.Input(type="number",id=f"sample-input-{i}",placeholder=f"{i}",min=0,className="mb-1") for (i,j) in zip(columns[1:],observation[1:])],
                                                                  className="d-flex flex-column justify-content-between"),
                                                                  dbc.Input(type="number",id="sample-input-OM",placeholder="OM",min=0,className=""),
                                                                  ],id="sample-input",className="mb-1")
sample_input_card = dbc.Card([dbc.CardBody(sample_input),dbc.CardBody([dbc.Button('Predict', id='predict-button')])],id="sample-input-card",color="light", outline=True)
parameter_tabs = dbc.Tabs(
    [
        dbc.Tab(knn_content, label="KNN"),
        dbc.Tab(dt_content, label="DECISON TREE"),
        dbc.Tab(rf_content, label="RANDOM FORESTS"),
    ]
)
####################################################### LAYOUT ####################################################################

layout = dbc.Container([
    html.H1("Classification"),
    html.Hr(),
    dcc.Store(id="current-model",data="",storage_type="session"),
    dbc.Row([
        dbc.Col(dbc.Card([html.H6("Choose an Algorithm and set its Parameters",className="pt-2 text-center"),parameter_tabs])),
        dbc.Col(dbc.Spinner(id="metrics-output"),className="fs-3"),
    ],className="mb-2"),
    dbc.Row([dbc.Col([dcc.Graph(id="confusion-matrix",)]),
             dbc.Col([dcc.Graph(id="true_pred_scatter")])],className="mt-1"),
    dbc.Row([dbc.Col([sample_input_card],width=3),dbc.Col([dbc.Card([dbc.CardHeader("Result")]),
                                                                                     dbc.CardBody(html.Div(id="prediction-output",className="text-center"))],className="p-5")],className="mt-2"),
    
],fluid=True,
className="dbc dbc-ag-grid")

####################################################### CALLBACK ####################################################################



@callback(
    [Output("current-model", "data"),Output("metrics-output", "children"),
    Output("confusion-matrix", "figure"),Output("true_pred_scatter", "figure")],
    [Input("knn-train-button", "n_clicks"),Input("rf-train-button", "n_clicks"),Input("dt-train-button", "n_clicks")],
    [State("knn-k", "value"),State("distance-type", "value"),
    State("dt-min-samples-split", "value"),State("dt-max-depth", "value"),
    State("rf-min-samples-split", "value"),State("rf-max-depth", "value"),State("n_trees", "value")]
)
def update_instnance_model(bt1,bt2,bt3,knn_k,distance_type,dt_min_samples_split,dt_max_depth,rf_min_samples_split,rf__max_depth,n_trees):
    
    if bt1 is None and bt2 is None and bt3 is None:
        model,predictions,metrics = train_predict(algorithme_type="KNN")
        fig_1 = confusion_matrix_plot(y_test_2, predictions)
        fig_2 = true_vs_predicted_labels_plot(y_test_2, predictions)
        return "knn_model.pkl",metrics, fig_1, fig_2
    if callback_context.triggered_id == "rf-train-button":
        model,predictions,metrics = train_predict(algorithme_type="RANDOM FORESTS",min_samples_split=rf_min_samples_split,max_depth=rf__max_depth,nb_trees=n_trees)
    elif callback_context.triggered_id == "knn-train-button":
        model, predictions, metrics = train_predict(algorithme_type="KNN",k=knn_k,distance_type=distance_type)
        fig_1 = confusion_matrix_plot(y_test_2, predictions)
        fig_2 = true_vs_predicted_labels_plot(y_test_2, predictions)
        return model, metrics, fig_1, fig_2
    elif callback_context.triggered_id == "dt-train-button":
        model,predictions, metrics = train_predict(algorithme_type="DECISON TREE",t_min_samples_split=dt_min_samples_split,t_max_depth=dt_max_depth)

    fig_1 = confusion_matrix_plot(y_test_1, predictions)
    fig_2 = true_vs_predicted_labels_plot(y_test_1, predictions)
    return model, metrics, fig_1, fig_2




@callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    [State("current-model", "data"),State("sample-input-N", "value"),
     State("sample-input-P", "value"),State("sample-input-K", "value"),State("sample-input-EC", "value"),
     State("sample-input-pH", "value"), State("sample-input-S", "value"),State("sample-input-Zn", "value"),
     State("sample-input-Fe", "value"),State("sample-input-Cu", "value"),State("sample-input-Mn", "value"),
     State("sample-input-B", "value"),State("sample-input-OM", "value")]
     )
def update_make_predcion(n_clicks,model_name,N,P,K,EC,pH,S,Zn,Fe,Cu,Mn,B,OM):
    if n_clicks is None:
        return None
    observation = [N,P,K,EC,pH,S,Zn,Fe,Cu,Mn,B,OM]
    prediction = make_prediction(model_name,observation)
    return prediction
    
   