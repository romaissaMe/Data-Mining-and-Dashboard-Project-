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

dash.register_page(__name__, path='/',name='Sol Database')
dataset_1 = pd.read_csv("./Dataset1.csv")
dataset_1['P'] = pd.to_numeric(dataset_1['P'], errors='coerce')
data = dataset_1
column_descriptions = pd.Series({
    'N':'Nitrogen content in soil',
    'P':'Phosphorus content in soil',
    'K': 'Potassium content in soil',
    'pH': 'Soil pH level',
    'EC': 'Electrical conductivity of the soil',
    'OC':'Organic carbon content in soil',
    'S':'Sulfur content in soil',
    'Zn':'Zinc content in soil',
    'Fe': 'Iron content in soil',
    'Cu': 'Copper content in soil',
    'Mn': 'Manganese content in soil',
    'B':'Boron content in soil',
    'OM': 'Organic matter content in soil',
    'Fertility': 'Fertility level (target variable)'
    
})


# Function to load dataset based on dropdown value
def load_dataset(selected_dataset):
    if selected_dataset == 'Before':
        dataset = pd.read_csv("./Dataset1.csv")
        dataset['P'] = pd.to_numeric(dataset_1['P'], errors='coerce')
        return dataset
    elif selected_dataset == 'After':
        dataset = pd.read_csv("./new_Dataset_1.csv")
        return dataset
dataset_type_dropdown = dcc.Dropdown(
    id='dataset-type-dropdown',
    value='Before',
    options=[
        {'label': 'Before Cleaning', 'value': 'Before'},
        {'label': 'After Cleaning', 'value': 'After'},
    ],
    className='database-dropdown mb-3 text-secondary bg-light',
)

attributes_dropdown = dcc.Dropdown(id='attributes-dropdown', className='attributes-dropdown text-secondary',options=list(column_descriptions.keys())[:-1],value='N')
scatter_plot_attributes_choice = dcc.Dropdown(id='scatter-plot-attributes-choice', className='attributes-dropdown-md text-secondary',options=list(column_descriptions.keys())[:-1],value=['N','P'],multi=True)
##################################################### Graphs Functions ########################################################

def fertility_features_bar_plot(data=data):
    grouped_data = data.groupby('Fertility').mean().reset_index()
    fig=px.bar(grouped_data, x='Fertility', y=[col for col in grouped_data.columns if col != 'Fertility'], barmode='group', 
                    labels={'Fertility': 'Fertility', 'value': 'Average Value', 'variable': 'Column'}
                   )
    return fig

def fertility_pie_plot(data=data):
    val=data['Fertility'].value_counts().values
    fig = go.Figure(data=[go.Pie(labels=[0,1,2], values=val, hole=0.7)])
    return fig

def correlation_matrix_plot(data=data):
        corr_matrix = data.corr()

        fig = px.imshow(corr_matrix,
                        )
        annotations = []
        for i, row in enumerate(corr_matrix.index):
            for j, col in enumerate(corr_matrix.columns):
                annotations.append(dict(text=str(round(corr_matrix.iloc[i, j], 1)),
                                        x=col, y=row, xref='x1', yref='y1',
                                        showarrow=False, font=dict(color='black', size=6.5)))
        
        fig.update_layout(annotations=annotations)
        return fig

def box_plot(data=data,attribute='N'):
    fig = px.box(data, y=attribute, x='Fertility')
    fig.update_layout(title=f"Boxplot for {attribute} Attribute")
    return fig

def histogram_plot(data=data,attribute='N'):
    fig = fig = px.histogram(data, x=attribute)
    fig.update_layout(yaxis_title='Count', xaxis_title=f"{attribute}", title=f"Histogram for {attribute} Attribute")
    
    return fig

def scatter_plot_attributes(data=data,x='N',y='P'):
    fig = px.scatter(data,y=y,x=x)
    fig.update_layout(title=f"Scatterplot {x} - {y}")
    return fig
    
def missing_values_kpi(data=data):
    total=data.isnull().sum()
    percent = (sum(total.values)/len(data))*100
    return percent
def size_data(data=data):
    return len(data)

def attribute_description(data=data,column='N'):
    description= [round(data[column].mean(),2),round(data[column].std(),2),round(data[column].min(),2),round(data[column].max(),2),round(data[column].quantile(0.25),2),round(data[column].quantile(0.5),2),round(data[column].quantile(0.75),2)]
    description_html = [
        html.Div([f"Mean: {description[0]}"], className="kpi-card purple"),
        html.Div([f"Std: {description[1]}"], className="kpi-card"),
        html.Div([f"Min: {description[2]}"], className="kpi-card purple"),
        html.Div([f"Max: {description[3]}"], className="kpi-card"),
        html.Div([f"25th Quantile: {description[4]}"], className="kpi-card  purple"),
        html.Div([f"Median: {description[5]}"], className="kpi-card"),
        html.Div([f"75th Quantile: {description[6]}"], className="kpi-card purple"),
    ]
    return description_html


layout = dbc.Container(children=[
    dataset_type_dropdown,
    dbc.Row(children=[
         dbc.Col(children=[
            dbc.Card([
                dbc.CardHeader('Average Values by Fertility'),
                dbc.CardBody([
                    dcc.Graph(id='fertility-features-bar-plot',className='graph-style')
                ],className='card-body-custom')
            ],)
         ],),
        dbc.Col(children=[
            dbc.Card([
                dbc.CardHeader('Correlation Matrix'),
                dbc.CardBody([
                    dcc.Graph(id='correlation-matrix-plot',className='graph-style')
                ],className='card-body-custom')
            ])
         ]),
        dbc.Col(children=[
            dbc.Card([
                dbc.CardHeader('Fertility Distribution'),
                dbc.CardBody([
                    dcc.Graph(id='fertility-pie-plot',className='graph-style')
                ],className='card-body-custom')
            ])
         ],width=2),
       ],className='mt-2'),
    dbc.Row([
    dbc.Col(id='attribute-description-output',style={'display': 'flex','flex-direction': 'row','wrap': 'none', 'justify-content': 'space-between', 'align-items': 'center', 'width': '100%',' height': '100%','margin-top': '2px'}),
    ],className='mt-2'),
    dbc.Row([dbc.Col([
        attributes_dropdown,
    ])],className="my-1"),
    dbc.Row(children=[
        dbc.Col([dcc.Graph(id='box-plot',)]),
        dbc.Col([dcc.Graph(id='histogram-plot')]),
    ],className='mt-2'),
     dbc.Row(children=[
        scatter_plot_attributes_choice,
        dbc.Col([dcc.Graph(id='scatter-plot')]),
    ],id='scatter-container',className='mt-2'),
    
],fluid=True,className='dbc dbc-ag-grid', id = 'container')




################################################# Callbacks to update layout components based on the selected database ############################################
@callback(
    [Output('attributes-dropdown', 'options'),Output('scatter-plot-attributes-choice', 'options')],
    [Input('dataset-type-dropdown', 'value')]
)
def update_dropdown(selected_data):
    data=load_dataset(selected_data)
    options = [i for i in data.columns if i != 'Fertility']
    scatter_plot_options = [i for i in data.columns if i != 'Fertility']
    return options,scatter_plot_options
@callback(
    Output('histogram-plot', 'figure'), 
    [Input('dataset-type-dropdown', 'value'),
     Input('attributes-dropdown', 'value')]
)
def update_histogram_plot(selected_data,attribute):
     data=load_dataset(selected_data)
     fig = histogram_plot(data,attribute)
     return fig

@callback(
    Output('box-plot', 'figure'), 
    [Input('dataset-type-dropdown', 'value'),
    Input('attributes-dropdown', 'value')]
)
def update_box_plot(selected_data,attribute):
     data=load_dataset(selected_data)
     fig = box_plot(data,attribute)
     return fig

@callback(
    Output('scatter-plot', 'figure'),
    [Input('dataset-type-dropdown', 'value'),
    Input('scatter-plot-attributes-choice', 'value')],
)
def update_scatter_plot(selected_data,attribute):
     data=load_dataset(selected_data)
     fig = scatter_plot_attributes(data,attribute[0],attribute[1])
     return fig

@callback(
    Output('correlation-matrix-plot', 'figure'),
    [Input('dataset-type-dropdown', 'value')],
)
def update_correlation_matrix_plot(selected_data):
     data=load_dataset(selected_data)
     fig = correlation_matrix_plot(data)
     return fig

@callback(
    Output('fertility-features-bar-plot', 'figure'),
    [Input('dataset-type-dropdown', 'value')]
)
def update_fertility_features_bar_plot(selected_data):
     data=load_dataset(selected_data)
     fig = fertility_features_bar_plot(data)
     return fig

@callback(
    Output('fertility-pie-plot', 'figure'),
    [Input('dataset-type-dropdown', 'value')],
)
def update_fertility_pie_plot(selected_data):
     data=load_dataset(selected_data)
     fig = fertility_pie_plot(data)
     return fig

@callback(
    Output('attribute-description-output', 'children'),
    [Input('dataset-type-dropdown', 'value'),Input('attributes-dropdown', 'value')]
)
def update_description(selected_data,selected_attribute):
    data = load_dataset(selected_data)
    if selected_attribute is None:
        return None  # If no attribute selected, don't update the description

    description = attribute_description(data, selected_attribute)
    return description
