import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
from flask import Flask
from investment_allocater import allocate_investment  # Import the investment allocation function

# Flask Server for Dash
server = Flask(__name__)
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True)

# Load Investment Data for Flask Visualization
file_path = "investment_allocation_companies.xlsx"
try:
    df_investment = pd.read_excel(file_path)
except FileNotFoundError:
    df_investment = pd.DataFrame(columns=["Sector", "Company", "20-29", "30-39", "40-49", "50-59", "60+"])

sectors = df_investment["Sector"].unique() if "Sector" in df_investment.columns else []

# Dash App Layout
app.layout = html.Div([
    html.H1(id="dashboard-title", style={
        'text-align': 'center',
        'font-size': '28px',
        'margin-bottom': '10px',
        'width': '100%',
        'position': 'absolute',
        'top': '0',
        'left': '50%',
        'transform': 'translateX(-50%)'
    }),

    html.Div(id="modal", children=[
        html.Div(className="popup-container", children=[
            html.H2("💲 Enter Your Details", className="popup-title"),
            dcc.Input(id="name-input", type="text", placeholder="Enter your name", className="popup-input"),
            dcc.Input(id="age-input", type="number", placeholder="Enter your age", min=20, max=65, className="popup-input"),
            dcc.Input(id="amount-input", type="number", placeholder="Enter amount", min=1000, className="popup-input"),
            html.Button("✔ Submit", id="submit-btn", n_clicks=0, className="popup-btn")
        ])
    ]),

    html.Div(id="dashboard-content", style={'display': 'none'}, children=[
        html.Div(className="dashboard-header", children=[
            html.Div(className="user-info", style={'display': 'flex', 'gap': '20px', 'align-items': 'center'},
                     children=[
                         html.P(id="user-age", className="user-detail",
                                style={'font-size': '18px', 'font-weight': 'bold', 'color': '#2C3E50'}),
                         html.P(id="user-amount", className="user-detail",
                                style={'font-size': '18px', 'font-weight': 'bold', 'color': '#27AE60'})
                     ]),
            html.Div(className="profile-info", style={'position': 'absolute', 'top': '10px', 'right': '20px'},
                     children=[
                         html.P(id="profile-type", className="profile-detail",
                                style={'font-size': '18px', 'font-weight': 'bold', 'color': '#E74C3C'})
                     ])
        ]),

        html.Div([dcc.Graph(id="investment-pie-chart")], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id="sector-dropdown",
                options=[{'label': sector, 'value': sector} for sector in sectors],
                placeholder="Select a Sector"
            ),
            dcc.Graph(id="sector-investment-bar-chart")
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Button("Download PDF", id="print-btn", className="print-button",
                    style={'display': 'block', 'margin': '20px auto', 'padding': '10px 20px', 'font-size': '16px',
                           'cursor': 'pointer'}),

        # Cluster Consistency Button
        html.Button("87% Cluster Consistency ⚡", id="cluster-btn", className="cluster-button")
    ])
])

# Callback to handle dashboard updates
@app.callback(
    [Output("dashboard-title", "children"),
     Output("investment-pie-chart", "figure"),
     Output("sector-investment-bar-chart", "figure"),
     Output("modal", "style"),
     Output("dashboard-content", "style"),
     Output("user-age", "children"),
     Output("user-amount", "children"),
     Output("profile-type", "children")],
    [Input("submit-btn", "n_clicks"), Input("sector-dropdown", "value")],
    [State("name-input", "value"), State("age-input", "value"), State("amount-input", "value")],
    prevent_initial_call=True
)
def update_dashboard(n_clicks, selected_sector, name, age, amount):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == "sector-dropdown":
        if not selected_sector or age is None or amount is None:
            return dash.no_update
        age_group = "20-29" if age < 30 else "30-39" if age < 40 else "40-49" if age < 50 else "50-59" if age < 60 else "60+"
        sector_data = df_investment[df_investment["Sector"] == selected_sector]
        sector_data[age_group] = (sector_data[age_group] / 100000) * amount
        fig_bar = px.bar(sector_data, x="Company", y=age_group, title=f"Investment Allocation in {selected_sector}",
                         labels={age_group: "Money Invested"})
        return dash.no_update, dash.no_update, fig_bar, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if triggered_id == "submit-btn":
        if not name or age is None or amount is None:
            return dash.no_update
        investment_recommendation = allocate_investment(age, amount)
        fig_pie = px.pie(names=list(investment_recommendation.keys()), values=list(investment_recommendation.values()),
                         title="Investment Allocation")
        profile_type = "Conservative" if age >= 60 else "Very Low Risk" if age >= 50 else "Low Risk" if age >= 40 else "Balanced" if age >= 30 else "Aggressive"
        return (
        f"{name}'s Dashboard 📊", fig_pie, dash.no_update, {'display': 'none'}, {'display': 'block'}, f"Age: {age}",
        f"Investment: ${amount}", f"Profile: {profile_type}")

    return dash.no_update
app.clientside_callback(
    """
    function(n) {
        if(n > 0) {
            window.print();  // Trigger browser's print dialog
        }
        return n;
    }
    """,
    Output("print-btn", "n_clicks"),
    Input("print-btn", "n_clicks")
)

if __name__ == "__main__":
    app.run(debug=True)
