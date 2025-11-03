import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Predictive Maintenance Dashboard'),
    dcc.Graph(id='shap-summary-plot'),
    dcc.Dropdown(
        id='instance-dropdown',
        options=[{'label': str(i), 'value': i} for i in range(len(X_test))],
        value=0
    ),
    dcc.Graph(id='lime-explanation')
])


@app.callback(Output('shap-summary-plot', 'figure'), [Input('instance-dropdown', 'value')])
def update_shap_plot(instance):
    shapsummary = go.Figure(shap.summary_plot(shap_values, X_test, plot_type='bar'))
    return shapsummary


@app.callback(Output('lime-explanation', 'figure'), [Input('instance-dropdown', 'value')])
def update_lime_plot(instance):
    explanation = lime_explainer.explain_instance(X_test[instance], xgb_model.predict, num_features=5)
    limeplot = explanation.as_pyplot_figure()
    return limeplot


if __name__ == '__main__':
    app.run_server(debug=True)
