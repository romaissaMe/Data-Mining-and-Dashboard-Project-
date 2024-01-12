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
from algorithms import apriori, generate_association_rules, recommandation_item

dash.register_page(__name__, path='/Frequent_Pattern_Mining',name='Frequent Patterns Mining')
data= pd.read_csv("./Dataset2.csv")
transactional_data = pd.read_csv("./dataitems.csv")
transactional_data ['Items'] = transactional_data ['Items'].apply(ast.literal_eval)
NB_FB = [(2, 1792), (3, 162), (4, 6), (5, 0), (6, 0), (7, 0)]

################################################## UTILITY FUNCTIONS #####################################################################

def nb_FP_per_Support(data=transactional_data, support=7):
    if support == 7:
        nb_FP = NB_FB
    else:
        nb_FP=[]
        for i in range(2,support+1):
            FP=apriori(data,i)
            nb_FP.append((i,len(FP)))
    support_levels = [x[0] for x in nb_FP]
    num_patterns = [x[1] for x in nb_FP]
    fig = px.line(x=support_levels, y=num_patterns, title="Number of Frequent Patterns for Different Support Levels") 
    return fig



def AR_per_confiance_level(dataitems=transactional_data):
    FP = apriori(dataitems,3)
    i=0.1
    nb_rules=[]
    while(i<1):
        association_rules=generate_association_rules(FP,i,dataitems)
        nb_rules.append((i,len(association_rules)))
        i+=0.1
    confiance_levels = [x[0] for x in nb_rules]
    num_rules = [x[1] for x in nb_rules]
    fig = px.line(x=confiance_levels, y=num_rules, title="Number of Association Rules for Different Confiance Levels")
    fig.update_layout(xaxis_title="Confiance Levels", yaxis_title="Number of Association Rules")
    return fig

def display_FP(data = transactional_data, support = 3):
    FP = apriori(data, support)
    return FP
  
def afficher_association_rules(association_rules, metric):
    printed_association_rules = []
    if metric  == "confidance" or metric == "":
        for rule in association_rules:
            item1,item2,support,confiance=rule
            printed_association_rules.append(f"{item1} -> {item2} ({support},{confiance})")
    else:
        for rule in association_rules:
            item_1, item_2 = rule
            item1,item2,support,confiance = item_1
            printed_association_rules.append(f"{item1} -> {item2} ({support},{confiance},{item_2})")
    return html.Ul([html.Li(printed_association_rules[i]) for i in range(len(printed_association_rules))]),
     

def display_lift_metric(association_rules, dataitems):
    forte_rules=[]
    for i in range(len(association_rules)):
        rule=association_rules[i]
        item1,item2,support,confiance=rule
        P_A = 0
        for index, row in dataitems.iterrows():
            if all(val in row['Items'] for val in item2):
                P_A += 1
        lift=round(confiance / P_A, 2)
        #lift=lift(rule)
        forte_rules.append((rule,lift))
        
    sorted(forte_rules, key=lambda x: x[1], reverse=True) # Sort forte_rules
    max_lift=max(forte_rules, key=lambda x: x[1])[1]
    forte_rules=[(rule,lift) for rule,lift in forte_rules if lift==max_lift]
    return forte_rules


def display_cosine_metric(association_rules, dataitems):
    forte_rules=[]
    for i in range(len(association_rules)):
        rule=association_rules[i]
        item1,item2,support,confiance=rule
        P_A_B = 0
        P_A = 0
        P_B = 0
        for index, row in dataitems.iterrows():
            if all(i in row['Items'] for i in item1) and all(i in row['Items'] for i in item2):
                P_A_B += 1
            if all(i in row['Items'] for i in item1):
                P_A += 1
            if all(i in row['Items'] for i in item2):
                P_B += 1

        cosine=round((P_A_B / math.sqrt(P_A*P_B)),2) if P_A != 0 else 0
        forte_rules.append((rule,cosine))

    sorted(forte_rules, key=lambda x: x[1], reverse=True) # Sort forte_rules
    max_cosine=max(forte_rules, key=lambda x: x[1])[1]
    forte_rules=[(rule,cosine) for rule,cosine in forte_rules if cosine==max_cosine]
    return forte_rules
    
def display_jaccard_metric(association_rules, dataitems):
    forte_rules=[]
    for i in range(len(association_rules)):
        rule=association_rules[i]
        item1,item2,support,confiance=rule
        P_A_B = 0
        P_A = 0
        P_B = 0
        for index, row in dataitems.iterrows():
            if all(i in row['Items'] for i in item1) and all(i in row['Items'] for i in item2):
                P_A_B += 1
            if all(i in row['Items'] for i in item1):
                P_A += 1
            if all(i in row['Items'] for i in item2):
                P_B += 1
        jaccard=round((P_A_B / (P_A + P_B - P_A_B)),2)
        forte_rules.append((rule,jaccard))

    sorted(forte_rules, key=lambda x: x[1], reverse=True) # Sort forte_rules
    max_jaccard=max(forte_rules, key=lambda x: x[1])[1]
    forte_rules=[(rule,jaccard) for rule,jaccard in forte_rules if jaccard==max_jaccard]
    return forte_rules


def display_kulczynski_metric(association_rules, dataitems):
    forte_rules=[]
    for i in range(len(association_rules)):
        rule=association_rules[i]
        item1,item2,support,confiance=rule
        P_A_B = 0
        P_A = 0
        P_B = 0
        for index, row in dataitems.iterrows():
            if all(i in row['Items'] for i in item1) and all(i in row['Items'] for i in item2):
                P_A_B += 1
            if all(i in row['Items'] for i in item1):
                P_A += 1
            if all(i in row['Items'] for i in item2):
                P_B += 1
        kulczynski= round(((1/2)*((P_A_B/P_A)+(P_A_B/P_B))),2)
        forte_rules.append((rule,kulczynski))

    sorted(forte_rules, key=lambda x: x[1], reverse=True) # Sort forte_rules
    max_kulczynski=max(forte_rules, key=lambda x: x[1])[1]
    forte_rules=[(rule,kulczynski) for rule,kulczynski in forte_rules if kulczynski==max_kulczynski]
    return forte_rules

def FP(data=transactional_data, support=3, metric_type="confidance",confidance_threshold= 0.7):
    FP = display_FP(data, support)
    FFP = generate_association_rules(FP, confidance_threshold, data)
    if metric_type == "" or metric_type == "confidance":
        return afficher_association_rules(FFP, metric_type), FFP
    elif metric_type == "lift":
        FA= display_lift_metric(FFP, data)
        return afficher_association_rules(FA, metric_type),FFP
    elif metric_type == "cosine":
        FA= display_cosine_metric(FFP, data)
        return afficher_association_rules(FA, metric_type),FFP
    elif metric_type == "jaccard":
        FA= display_jaccard_metric(FFP, data)
        return afficher_association_rules(FA, metric_type),FFP
    elif metric_type == "kulczynski":
        FA= display_kulczynski_metric(FFP, data)
        return afficher_association_rules(FA, metric_type),FFP

def display_recommandedation(RA,observation = (2.0, 3.0, 2.0, 'Coconut', 'DAP')):
    result = recommandation_item(observation,RA)
    return result
   

############################################################ Layout Components ####################################################################
metrics_checklist = html.Div(
    [
        dbc.Label("Select Metric"),
        dcc.Dropdown(
            id="metrics",
            options=[
                {"label": "Confidance", "value": "confidance"},
                {"label": "Lift", "value": "lift"},
                {"label": "Cosine", "value": "cosine"},
                {"label": "Jaccard", "value": "jaccard"},
                {"label": "Kulczynski", "value": "kulczynski"},
            ],
            value="confidance",
           className="text-primary",
        ),
        
    ],
)

support_slider = html.Div(
    [
        dbc.Label("Minimum Support"),
        dcc.Slider(
            id="support",
            min=1,
            max=10,
            step=1,
            value=3,
            marks=None,
            tooltip={"placement": "bottom", "always_visible": True},
        )
    ]
)
confidance_slider = html.Div(
    [
        dbc.Label("Minimum Confidance"),
        dcc.Slider(
            id= "confidance",
            min=0,
            max=1,
            step=0.1,
            value=0.7,
            marks=None,
            tooltip={"placement": "bottom", "always_visible": True},
        )
    ]
)
controls = dbc.Card(
    [support_slider, confidance_slider, metrics_checklist],
    body=True,
)

transactional_data_grid = dag.AgGrid(
    id="transactional-data",
    columnDefs=[{"field": i} for i in transactional_data.columns],
    rowData=transactional_data.to_dict("records"),
    defaultColDef={"flex": 1, "minWidth": 120, "sortable": True, "resizable": True, "filter": True},
    dashGridOptions={"rowSelection":"multiple"},
    columnSize="sizeToFit",
)

temperature_input = dbc.Input(
    type="number",
    id="temperature",
    value = 2.0,
    min=10,
    max=30,
    className="me-1"

)
humidity_input = dbc.Input(
    type="number",
    id="humidity",
    value=3.0,
    min=float(10),
    max=float(30),
    className="me-1"
)
rainfall_input = dbc.Input(
    type="number",
    id="rainfall",
    value=2.0,
    min=10.0,
    max=30.0,
)
crop_input = dcc.Dropdown(
   options=["Coconut","DAP"],
   id="crop",
   value="Coconut",
   clearable=False,
   className="mb-2"
)
fertilizer_input = dcc.Dropdown(
   options=["DAP","Urea","MOP","Good NPK"],
   id = "fertilizer",
   value="DAP",
   clearable=False,
   className="mb-2"
)

recommandation_btn = dbc.Button("Search", id="recommandation-btn")
attributes_1 = html.Div(
    [
       temperature_input,humidity_input,rainfall_input, 
    ],
    className="mb-2 d-flex justify-content-between",
)

recommandation_options = dbc.Card([attributes_1,crop_input,fertilizer_input,recommandation_btn],className="p-2")
################################################################ Page Layout ################################################################

layout= dbc.Container([
   
    html.H4("Association Rules"),
    html.Hr(className="mt-1"),
    dcc.Store(id="current-RA",data="",storage_type="session"),
    dbc.Row([
        dbc.Col(
            [
            transactional_data_grid
            ]
        ),
        dbc.Col(
            [
                dcc.Graph(id='nb-FP-support',figure=nb_FP_per_Support())
            ]
        ),
        dbc.Col(
            [
                dcc.Graph(id='AR-confiance',figure=AR_per_confiance_level())
            ]
        ),
        
    ],className="mb-3"),
    dbc.Row([
        dbc.Col(
            [   
               controls, 
            ],
            width=4
        ),
      dbc.Col(children=[html.H5("FP Space"),html.Div(id="FP-space",className="scrollable-div")])
    ]),
    
    dbc.Row([dbc.Col([html.H5("Try The Recommandation System"),recommandation_options],width=4),
             dbc.Col([
                 dbc.CardHeader("Recommandation Result"),
                 dbc.CardBody([
                 ],id="recommandation-soil",)
             ],className="p-3")
             ],
            )],
    fluid=True,
    className="dbc dbc-ag-grid")


##################################################### Callbacks ##################################################################################################
@callback(
    [Output("FP-space","children"),Output("current-RA","data")],
    [Input("support","value"),Input("metrics","value"),Input("confidance","value")]
)
def update_FP(selected_support,selected_metric,selected_confidance):
    FR, RA = FP(transactional_data,selected_support,selected_metric,selected_confidance)
    return FR, str(RA)

@callback(
    Output("recommandation-soil","children"),
    Input("recommandation-btn","n_clicks"),
    [State("temperature","value"),State("humidity","value"),State("rainfall","value"),State("crop","value"),
     State("fertilizer","value"),State("current-RA","data")]
)
def update_recommandation(n_clicks,slected_temperature,selected_humidity,selected_rainfall,selected_crop,selected_fertilizer,RA):
    observation = (slected_temperature,selected_humidity,selected_rainfall,selected_crop,selected_fertilizer)
    try:
        RA_list= ast.literal_eval(RA)
        return display_recommandedation(RA_list,observation)
    except SyntaxError as e:
        print(f"SyntaxError: {e}")
        return [" "]
    
    
