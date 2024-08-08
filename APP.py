import dash
from dash import dcc, html, Input, Output, State
import dash_table
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import base64
import io

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Sales Performance Dashboard"),
    
    # File upload
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    
    # Data table
    dash_table.DataTable(id='data-table'),
    
    # Dropdowns and filters
    dcc.DatePickerRange(id='date-picker'),
    dcc.Dropdown(id='category-dropdown', placeholder="Select a Category"),
    
    # Graphs and plots
    dcc.Graph(id='time-series-plot'),
    dcc.Graph(id='histogram-plot'),
    dcc.Graph(id='lm-plot'),
    
    # Export button
    html.Button("Download Filtered Data", id="download-button"),
    dcc.Download(id="download-dataframe-csv")
])

# Helper function to parse the uploaded file
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return df

@app.callback(
    Output('data-table', 'data'),
    Output('data-table', 'columns'),
    Output('date-picker', 'start_date'),
    Output('date-picker', 'end_date'),
    Output('category-dropdown', 'options'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_table(contents, filename):
    if contents is None:
        return [], [], None, None, []
    
    df = parse_contents(contents, filename)
    columns = [{"name": col, "id": col} for col in df.columns]
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    options = [{'label': cat, 'value': cat} for cat in df['Category'].unique()]
    
    return df.to_dict('records'), columns, start_date, end_date, options

@app.callback(
    Output('time-series-plot', 'figure'),
    Output('histogram-plot', 'figure'),
    Output('lm-plot', 'figure'),
    Input('data-table', 'data'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date'),
    Input('category-dropdown', 'value')
)
def update_plots(data, start_date, end_date, category):
    df = pd.DataFrame(data)
    
    if not df.empty:
        # Apply filters
        if start_date and end_date:
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        if category:
            df = df[df['Category'] == category]

        # Time series plot
        time_series_fig = px.line(df, x='Date', y='Sales', title='Sales Over Time')

        # Histogram plot
        histogram_fig = px.histogram(df, x='Sales', nbins=20, title='Sales Distribution')

        # Predictive model (linear regression)
        model = LinearRegression()
        df['DateOrdinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)
        X = df[['DateOrdinal']]
        y = df['Sales']
        model.fit(X, y)
        df['Prediction'] = model.predict(X)
        lm_fig = px.scatter(df, x='Date', y='Sales', title='Sales Forecast')
        lm_fig.add_traces(px.line(df, x='Date', y='Prediction').data)
        
        return time_series_fig, histogram_fig, lm_fig
    else:
        return {}, {}, {}

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-button", "n_clicks"),
    State('data-table', 'data'),
    prevent_initial_call=True
)
def download_filtered_data(n_clicks, data):
    df = pd.DataFrame(data)
    return dcc.send_data_frame(df.to_csv, "filtered_sales_data.csv")

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
