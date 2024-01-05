import dash
from dash import Dash, html, dash_table, dcc, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

load_figure_template(['vapor_dark','united_dark'])
stylesheets= ['./assets/style.css',dbc.themes.VAPOR]
app = Dash(__name__, suppress_callback_exceptions=True, pages_folder='pages', use_pages=True, external_stylesheets=stylesheets)


dataset_dropdown = dcc.Dropdown(
    id='dataset-dropdown',
    value='Sol Database',
    options=[
        {'label': 'Sol Database', 'value': 'Sol Database'},
        {'label': 'Covid Database', 'value': 'Covid Database'},
        {'label': 'Climate Database', 'value': 'Climate Database'},
    ],
    className='database-dropdown',
)

app.layout = html.Div(children=[
    html.Div(children=[
        dcc.Link(
            page['name'],href=page['relative_path']
        ) for page in dash.page_registry.values()
    ],id='page-content'),
    html.Div([dash.page_container],id='page-container',className='page-container'),
    
],className='main-container',id='main-container')



if __name__ == '__main__':
    app.run_server(debug=True)
    

