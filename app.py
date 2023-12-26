import dash
from dash import Dash, html, dash_table, dcc, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=['./assets/style.css'])
df = pd.read_csv("./Dataset1.csv")
df['P'] = pd.to_numeric(df['P'], errors='coerce')
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





app.layout = html.Div([
    html.H1('Data Exploration',className='title'),
    dcc.Dropdown(
        options=[
            {'label': 'Sol Database', 'value': 'sol'},
            {'label': 'Covid Database', 'value': 'covid'},
            {'label': 'Climate Database', 'value': 'climate'}
        ],
        value='sol',
        id='database-dropdown',
        className='database-dropdown'
    ),
    html.Div([
           dcc.Graph(id='fertility_features_bar_plot'),
           # Create a table to display column names and their descriptions
           html.Div([html.H4('Data Description'),
                     html.Table(id='data-description')
                     ]),
           
           html.Div([html.H4('Fertility Distribution'),
                   dcc.Graph(id='fertility_distribution_pie'),
                   ])
       ],style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'}),

    dcc.Dropdown(value='N',id='column-dropdown'),
    # Graph components
    dcc.Graph(id='histogram'),
    dcc.Graph(id='scatter-plot'),
    html.Div([
        html.H3('Row Data'),
        dash_table.DataTable(id='row-data',page_size=10),
    ])
])




# Callbacks to update layout components based on the selected database
@app.callback(
    [
    Output('data-description', 'data'),
    Output('fertility_distribution_pie', 'figure'),
    Output('column-dropdown', 'options'),
    ],
    Input('database-dropdown', 'value')
)
def update_content(selected_database):
    if selected_database == 'sol':
        dataset_info = infoOut(df,description=column_descriptions)
        dataset_description =dataset_info.to_dict('records')
        # Group by 'Fertility' and calculate the mean of each column
        grouped_data = df.groupby('Fertility').mean().reset_index()
        fertility_counts = df['Fertility'].value_counts()
        #pie chart and bar chart
        bar_plot = px.bar(grouped_data, x='Fertility', y=[col for col in grouped_data.columns if col != 'Fertility'], barmode='group', 
                    labels={'Fertility': 'Fertility', 'value': 'Average Value', 'variable': 'Column'},
                    title='Average Values by Fertility')
        fertility_distribution_pie = go.Figure(data=[go.Pie(labels=[0,1,2], values=[442, 402,  41], hole=0.3)])
        column_dropdow_options = [{'label':col,'value':col} for col in df.columns]
        # data_table = df.to_dict('records')
        table_header = [
        html.Tr([html.Th(col) for col in dataset_info.columns])
        ]
        table_rows = [
            html.Tr([
                html.Td(row[col]) for col in dataset_info.columns
            ]) for row in dataset_info
        ]
        table_content = table_header + table_rows

    return  table_content,fertility_distribution_pie, column_dropdow_options 
           


# Callback to update graphs based on dropdown selection
@app.callback(
    [dash.dependencies.Output('histogram', 'figure'),
     dash.dependencies.Output('scatter-plot', 'figure')],
    [dash.dependencies.Input('column-dropdown', 'value')]
)
def update_graphs(selected_column):
    # Create histogram
    histogram = px.histogram(df, x=selected_column, title=f'Histogram of {selected_column}')

    # Create scatter plot (against Fertility column)
    scatter_plot = px.scatter(df, x=selected_column, y='Fertility', title=f'Scatter Plot: {selected_column} vs Fertility')

    return histogram, scatter_plot


#utility functions
def infoOut(data,description,details=False):
    dfInfo = data.columns.to_frame(name='Column')
    dfInfo['Non-Null Count'] = data.notna().sum()
    dfInfo['Dtype'] = data.dtypes
    dfInfo['Description'] = description
    dfInfo.reset_index(drop=True,inplace=True)
    if details:
        rangeIndex = (dfInfo['Non-Null Count'].min(),dfInfo['Non-Null Count'].min())
        totalColumns = dfInfo['Column'].count()
        dtypesCount = dfInfo['Dtype'].value_counts()
        totalMemory = dfInfo.memory_usage().sum()
        return dfInfo, rangeIndex, totalColumns, dtypesCount, totalMemory
    else:
        return dfInfo

if __name__ == '__main__':
    app.run_server(debug=True)
    
