import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pickle
import numpy as np
from sklearn.decomposition import PCA

# Load the CSV data (ensure the correct path to your CSV)
input_csv_path = 'financial_portfolios.csv'
output_csv_path = 'data.csv'
kmeans_model_path = 'trained_kmeans_model.pkl'

# Read input data
input_df = pd.read_csv(input_csv_path)

# Rename columns to match expected names
column_mapping = {
    'Sector_Balance (%)': 'Sector_Balance',
    'Asset_Class_Distribution (%)': 'Asset_Class_Distribution',
    'Regional_Exposure (%)': 'Regional_Exposure'
}
input_df.rename(columns=column_mapping, inplace=True)

# Load the .pkl file and inspect its structure
with open(kmeans_model_path, 'rb') as file:
    model_data = pickle.load(file)

# Debugging: Print the structure of the .pkl file
print("Contents of the .pkl file:", model_data)

# Extract required columns from the .pkl file
if isinstance(model_data, (list, np.ndarray)):
    required_columns = list(model_data)
else:
    raise ValueError("The .pkl file must contain a list or array of required column names.")

# Check if the input CSV contains all required columns
if not all(col in input_df.columns for col in required_columns):
    raise ValueError(f"Input CSV must contain the following columns: {', '.join(required_columns)}")

# Derive clusters using a basic logic (e.g., k-means approximation)
def assign_basic_clusters(df, columns, n_clusters=3):
    # Use mean values to segment data into clusters
    centroids = [df[columns].mean() + (i - n_clusters // 2) * df[columns].std() for i in range(n_clusters)]

    def assign_cluster(row):
        distances = [np.linalg.norm(row[columns] - centroid) for centroid in centroids]
        return np.argmin(distances)

    return df.apply(assign_cluster, axis=1)

# Apply clustering to the input data
input_df['Cluster_Label'] = assign_basic_clusters(input_df, required_columns)

# Perform PCA for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(input_df[required_columns])
input_df['PCA1'] = pca_components[:, 0]
input_df['PCA2'] = pca_components[:, 1]

# Save the processed data to output CSV
input_df.to_csv(output_csv_path, index=False)

# Load the processed data into the Dash app
df = pd.read_csv(output_csv_path)

# Preprocess the 'Asset_Class_Distribution' column:
df['Asset_Class_Distribution'] = df['Asset_Class_Distribution'].fillna('').apply(lambda x: str(x).split(','))
df_exploded = df.explode('Asset_Class_Distribution')

# Calculate the Total Asset Value (sum of Sector Balances)
total_asset_value = df['Sector_Balance'].sum()

# Pie chart: Sector Balance
fig_pie = px.pie(df, names='Cluster_Label', values='Sector_Balance', title='Sector Balance Distribution')

# Line plot: Cluster Label vs Regional Exposure
fig_line = go.Figure()
fig_line.add_trace(
    go.Scatter(x=df['Cluster_Label'], y=df['Regional_Exposure'], mode='lines+markers', name='Regional Exposure'))
fig_line.update_layout(title='Cluster Label vs Regional Exposure', xaxis_title='Cluster Label',
                       yaxis_title='Regional Exposure')

# Line plot: Cluster Label vs Number of Elements in Cluster
elements_in_cluster = df.groupby('Cluster_Label').size()
fig_elements_cluster = go.Figure()
fig_elements_cluster.add_trace(
    go.Scatter(x=elements_in_cluster.index, y=elements_in_cluster.values, mode='lines+markers', name='Element Count'))
fig_elements_cluster.update_layout(title='Cluster Label vs Number of Elements in Cluster',
                                   xaxis_title='Cluster Label', yaxis_title='Number of Elements')

# PCA scatter plot for clustering visualization
fig_pca = px.scatter(
    df, x='PCA1', y='PCA2', color='Cluster_Label', title='Portfolio Clustering',
    labels={'PCA1': 'PCA Component 1', 'PCA2': 'PCA Component 2'},
    color_continuous_scale=px.colors.sequential.Viridis
)

# Set default figure for the bar chart
fig_bar = px.bar(df_exploded, x='Asset_Class_Distribution', title="Asset Class Distribution",
                 labels={"Asset_Class_Distribution": "Asset Class Value"}, category_orders={
        'Asset_Class_Distribution': sorted(df_exploded['Asset_Class_Distribution'].unique())
    })

# Create a Dash application
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Data Visualization Dashboard", className='dashboard-title'),
    html.Title("Dashboard"),
    # Total Asset Value as a Button at Top Left
    html.Div(f"Total Asset Value: ${total_asset_value:,.2f}",
             className='total-asset-value-button'),

    # Flex container for graphs
    html.Div([  # Flex container for graphs
        html.Div([  # Pie chart
            dcc.Graph(figure=fig_pie, className='dash-graph')
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        html.Div([  # Line plot for Cluster Label vs Regional Exposure
            dcc.Graph(figure=fig_line, className='dash-graph')
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '40px'}),

    # Cluster Label vs Number of Elements in Cluster plot
    html.Div([  # Elements in Cluster line plot
        dcc.Graph(figure=fig_elements_cluster, className='dash-graph')
    ], style={'marginBottom': '40px'}),

    # PCA scatter plot for clustering visualization
    html.Div([
        dcc.Graph(figure=fig_pca, className='dash-graph')
    ], style={'marginBottom': '40px'}),

    # Flexbox layout for the last graph (Bar Chart) and the Dropdown
    html.Div([
        # Bar Plot for Asset Class Distribution Values
        html.Div([  # Bar plot of Asset Class Distribution values
            dcc.Graph(id='bar-plot', figure=fig_bar)
        ], style={'flex': 1, 'marginRight': '20px'}),

        # Dropdown for selecting Cluster for Bar Chart
        html.Div([
            html.Label('Select Cluster for Asset Class Distribution Bar Chart'),
            dcc.Dropdown(
                id='cluster-dropdown',
                options=[{'label': cluster, 'value': cluster} for cluster in df['Cluster_Label'].unique()],
                value=df['Cluster_Label'].iloc[0],  # Default value (first cluster in the list)
                style={'width': '100%'}
            ),
        ], style={'flex': 0.4, 'marginTop': '20px'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'flex-start'}),

    # Generate Report Button
    html.Div([
        html.Button("Generate Report", id="generate-report-button", className="generate-report-button"),
        dcc.Download(id="download-pdf")
    ], style={'marginTop': '40px', 'textAlign': 'center'})
])

# Callback to update the bar chart based on selected cluster
@app.callback(
    Output('bar-plot', 'figure'),
    Input('cluster-dropdown', 'value')
)
def update_bar_chart(selected_cluster):
    # Filter the data based on the selected cluster
    filtered_data = df_exploded[df_exploded['Cluster_Label'] == selected_cluster]

    # Count the occurrences of each asset class for the selected cluster
    asset_class_counts = filtered_data['Asset_Class_Distribution'].value_counts().reset_index()
    asset_class_counts.columns = ['Asset_Class_Distribution', 'Count']

    # Create a bar chart for the selected cluster
    fig_bar_selected = px.bar(asset_class_counts, x='Asset_Class_Distribution', y='Count',
                              title=f"Asset Class Distribution values for Cluster {selected_cluster}",
                              labels={"Asset_Class_Distribution": "Asset Class Value", "Count": "Count"},
                              category_orders={'Asset_Class_Distribution': sorted(
                                  asset_class_counts['Asset_Class_Distribution'].unique())})

    return fig_bar_selected

if __name__ == '__main__':
    app.run_server(debug=True)
