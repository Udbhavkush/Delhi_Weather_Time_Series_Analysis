import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px

# Load time series data
df = pd.read_csv("dash_data.csv")

# Create Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div(children=[
    html.H1(children='Time Series Dashboard'),

    # Dropdown to select time series column
    html.Label("Select a time series column:"),
    dcc.Dropdown(
        id="col-select",
        options=[{"label": col, "value": col} for col in df.columns],
        value=df.columns[0]
    ),

    # Line chart to display time series
    dcc.Graph(id='time-series-graph'),

    # Date range picker to select time range
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=df['date'].min(),
        end_date=df['date'].max()
    ),

    # Button to reset date range picker
    html.Button('Reset Date Range', id='reset-button'),

    # Summary statistics of time series
    html.H2("Summary Statistics"),
    html.Div(id="stats-div"),
])


# Define callback for updating line chart
@app.callback(
    dash.dependencies.Output('time-series-graph', 'figure'),
    dash.dependencies.Input('col-select', 'value'),
    dash.dependencies.Input('date-picker-range', 'start_date'),
    dash.dependencies.Input('date-picker-range', 'end_date'))
def update_time_series(col_select, start_date, end_date):
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    fig = px.line(filtered_df, x="date", y=col_select)
    fig.update_layout(title=col_select + " Time Series")
    return fig


# Define callback for resetting date range picker
@app.callback(
    dash.dependencies.Output('date-picker-range', 'start_date'),
    dash.dependencies.Output('date-picker-range', 'end_date'),
    dash.dependencies.Input('reset-button', 'n_clicks'))
def reset_date_range(n_clicks):
    return df['date'].min(), df['date'].max()


# Define callback for updating summary statistics
@app.callback(
    dash.dependencies.Output('stats-div', 'children'),
    dash.dependencies.Input('col-select', 'value'),
    dash.dependencies.Input('date-picker-range', 'start_date'),
    dash.dependencies.Input('date-picker-range', 'end_date'))
def update_stats(col_select, start_date, end_date):
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    stats = filtered_df[col_select].describe().to_frame().reset_index().rename(
        columns={"index": "Statistic", col_select: "Value"})
    stats_html = stats.to_html(index=False)
    return html.Table(
        [html.Tr([html.Th(col) for col in stats.columns])] + [html.Tr([html.Td(val) for val in row]) for row in
                                                              stats.values])


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
    # app.run_server(debug=True, host="0.0.0.0", port=8080)