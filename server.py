# Import packages
from dash import Dash, html, dcc, callback, Output, Input, ctx, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import shapely
import numpy as np

# Global Variables
df = pd.read_csv('experiments/results.csv')
categories = ["Similarity", "Coverage", "Runtime"] # Metrics for Radar Chart
methods_list = ['ar', 'cchvae', 'cem', 'clue', "cruds", 'dice', "face_knn", "face_epsilon", "gs", "mace", "revise", "wachter"] # List of Available Recourse Methods for Radar Chart
l_norms_list = ['L0_distance', 'L1_distance', 'L2_distance', 'Linf_distance'] # List of Different L Norms for Radar Chart

def generate_best_methods(dataset, ml_model, l_norm):
  """
  Returns normalized data across three metrics (coverage, runtime, similarity)
  for each recourse model that is benchmarked using dataset and ml_model using the associated l_norm

  Parameters
  ----------
  dataset : str
    Dataset to consider from the benchamrk results.
  ml_model : str
      Model to consider from the benchamrk results.
  l_norm : str
      The l_norm to utilize from the benchmark results.
      
  Returns
  -------
  metrics_dict :  dict
      A dictionary that contains the normalized data across three metrics for
      each recourse method.
  """
  methods_dict = {}
  methods_values = []

  for method in methods_list:
    metric_list = []

    similarity_average = df.loc[(df['Dataset'] == dataset) & (df['ML_Model'] == ml_model) & (df['Recourse_Method'] == method)].values[: , l_norm]
    similarity_average = np.nanmean(((similarity_average)), dtype='float32')
    metric_list.append(similarity_average)

    coverage_average = df.loc[(df['Dataset'] == dataset) & (df['ML_Model'] == ml_model) & (df['Recourse_Method'] == method)].values[: , -2]
    coverage_average = np.nanmean(((coverage_average)), dtype='float32')
    metric_list.append(coverage_average)

    runtime_average = df.loc[(df['Dataset'] == dataset) & (df['ML_Model'] == ml_model) & (df['Recourse_Method'] == method)].values[: , -1]
    runtime_average = np.nanmean(((runtime_average)), dtype='float32')
    metric_list.append(runtime_average)

    methods_values.append(metric_list)

  # Convert to NP Array Multi-Dimensional Slicing
  methods_values = np.array(methods_values)

  # Normalize averages between datasets for Similarity, and Runtime
  for col in range(len(methods_values[0])):
    col_values = methods_values[:, col]
    col_values[np.isnan(col_values)] = np.nanmax(col_values) if col != 1 else 0 #Convert nan to the max value for similarity and runtime or 0 for coverage
    col_values = 1 - (col_values/np.nanmax(col_values)) if col != 1 else (col_values)
    methods_values[:, col] = col_values

  for i in range(len(methods_list)):
    method = methods_list[i]
    methods_dict[method] = methods_values[i]

  return methods_dict

def extract_trace_area(figure_data):
  """
  Returns the area for each trace in the radar chart

  Parameters
  ----------
  figure_data : pd.dataframe
    Dataset to consider from the benchamrk results.
      
  Returns
  -------
  df_a :  dict
      Area of each trace in the radar chart.
  """
  # get data back out of figure
  df = pd.concat(
      [
          pd.DataFrame({"r": t.r, "theta": t.theta, "trace": np.full(len(t.r), t.name)})
          for t in figure_data
      ]
  )
  # convert theta to be in radians
  df["theta_n"] = pd.factorize(df["theta"])[0]
  df["theta_radian"] = (df["theta_n"] / (df["theta_n"].max() + 1)) * 2 * np.pi
  # work out x,y co-ordinates
  df["x"] = np.cos(df["theta_radian"]) * df["r"]
  df["y"] = np.sin(df["theta_radian"]) * df["r"]

  # now generate a polygon from co-ordinates using shapely
  # then it's a simple case of getting the area of the polygon

  df_a = df.groupby("trace").apply(
      lambda d: shapely.geometry.MultiPoint(list(zip(d["x"], d["y"]))).convex_hull.area
  )

  return df_a

# design of the modal
modal = html.Div(
  [
    dbc.Button(
      "CARLA Benchmark Command",
      id="open",
      n_clicks=0,
      style={'font-size': '15px', 'color': '#000000', 'background-color': '#FFFFFF', 'border-color': '#808080', 'border-width': '1'}
    ),
    dbc.Modal(
      [
        dbc.ModalHeader(dbc.ModalTitle("CARLA Benchmark Command"), style={'font-size': '15px'}),
        dbc.ModalBody(children=[
          dcc.Markdown('''
            `def sum(a, b)`
          ''', id="command-markdown-id", style={'background-color': '#F2F2F2', 'font-size': '15px', 'padding': '3px 40px 30px 15px'}),
          dcc.Clipboard(
            target_id="command-markdown-id",
            style={
              "position": "absolute",
              "top": 15,
              "right": 15,
              "fontSize": 15,
            },
          ),
        ]),
      ],
      id="modal",
      is_open=False,
      centered=True
    ),
  ])

# Initialize the app - incorporate css
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# App layout
app.layout = html.Div([
    html.Div(children='My First App with Data, Graph, and Controls'),
    html.Hr(),  
    html.Div([
      html.Div([
        dcc.Dropdown(['All Available Recourse Models', 'Top 3 Recourse Models', 'Top 4 Recourse Models', 'Categorical Data'], 'All Available Recourse Models', id='constraints-dropdown'),
      ], style={'width': '20%'}),
      html.Div([
        dcc.Dropdown(['adult', 'compass', 'credit'], 'adult', id='dataset-radar-dropdown'),
      ], style={'width': '20%'}),
      html.Div([
        dcc.Dropdown(['mlp', 'linear'], 'mlp', id='ml-model-radar-dropdown')
      ], style={'width': '20%'}),
      html.Div([
        modal
      ], style={'width': '20%'}),
    ], style={'display': 'flex', 'gap': '10px'}),
    html.Hr(),
    html.Div([dcc.Graph(figure={}, id=f'radar-graph-{method}') for method in l_norms_list], style={'display': 'grid', 'grid-template-columns': 'repeat(2, 1fr)', 'grid-gap': '.75rem 1rem'}),
], style={'font-size': "15px", 'padding': '10px'})

# Add controls to build the interaction
@callback(
    Output(component_id='radar-graph-L0_distance', component_property='figure'),
    Output(component_id='radar-graph-L1_distance', component_property='figure'),
    Output(component_id='radar-graph-L2_distance', component_property='figure'),
    Output(component_id='radar-graph-Linf_distance', component_property='figure'),
    Input(component_id='dataset-radar-dropdown', component_property='value'),
    Input(component_id='ml-model-radar-dropdown', component_property='value'),
    Input(component_id='constraints-dropdown', component_property='value')
)
def update_graph(dataset, ml_model, top_val_constraints):
  """
  Updates Radar Chart when dropdown inputs change

  Parameters
  ----------
  Dataset Dropdown : Input
    Dataset Dropdown Inputs.
  Model Dropdown : Input
    Model Dropdown Inputs.
  Constraints Dropdown : Input
    Constraints Dropdown Inputs.
         
  Returns
  -------
  radar_graphs: List
    A list of updated radar graphs
  """
  l_norm_to_category_column_dict = {
    "L0_distance": 3,
    "L1_distance": 4,
    "L2_distance": 5,
    "Linf_distance": 6
  }

  val_constraints_to_num_dict = {
    'Top 3 Recourse Models': 3,
    'Top 4 Recourse Models': 4,
  }

  fig = []

  for norm_label, norm_value in l_norm_to_category_column_dict.items():
    methods_info = generate_best_methods(dataset, ml_model, norm_value)
    data = [go.Scatterpolar(
      r=row,
      theta=categories,
      fill='toself',
      name=label) for label, row in methods_info.items()
    ]
    
    curr_fig = go.Figure(
      data=data,
      layout=go.Layout(
        title=go.layout.Title(text=f'Chart for {norm_label}', xanchor='center', x=0.5),
        polar = dict(
          bgcolor = "white",
          angularaxis = dict(
            gridwidth = 1,
            gridcolor = "black"
          ),
          radialaxis = dict(
            linewidth = 1,
            gridcolor = "black",
            linecolor='black',
            gridwidth = 1,
            visible = True,
          )
        ),
        showlegend=True
      )
    )

    trace_area = extract_trace_area(curr_fig.data)
    trace_area_df = pd.DataFrame({'Trace':trace_area.index, 'Area':trace_area.values})
    

    # Sort in order of decreasing area
    trace_area_df = trace_area_df.sort_values(by=['Area'], ascending=False)

    # Get the indices of the sorted DataFrame
    sorted_indices = trace_area_df.index

    # Rearrange the list of objects based on the sorted DataFrame
    sorted_list_of_objects = [curr_fig.data[i] for i in sorted_indices]

    # Choose the top values based on the passed criteria option
    val_constraint = len(sorted_list_of_objects)-1
    if top_val_constraints in val_constraints_to_num_dict:
      val_constraint = val_constraints_to_num_dict[top_val_constraints]
    curr_fig.data = sorted_list_of_objects[:val_constraint]

    curr_fig.for_each_trace(lambda t: t.update(name=f"{t.name}, Area: {trace_area.loc[t.name]:.5f}"))

    fig.append(curr_fig)

  return fig

@callback(
  [
      Output("modal", "is_open"),
      Output(component_id='command-markdown-id', component_property='children')
  ],
  [
    Input("open", "n_clicks"),
    Input(component_id='ml-model-radar-dropdown', component_property='value'),
    Input(component_id='dataset-radar-dropdown', component_property='value'),
  ],
  [
      State("modal", "is_open"),
  ],
  prevent_initial_call=True
)
def toggle_modal(open_modal, chosen_model, chosen_dataset, is_open):
  """
  Handles toggling the benchmark tool modal, based on the chosen dropdown

  Parameters
  ----------
  Dataset Dropdown : Input
    Dataset Dropdown Inputs.
  Model Dropdown : Input
    Model Dropdown Inputs.
  Button Clicks : Input
    Button Click Inputs.
  Modal State : State
    State of the modal (Open vs Close).
         
  Returns
  -------
  modalInfo: List
    A list of the state of the modal (Open vs Close), and the information to show in the Modal.
  """
  # which button triggered the callback?
  trigger = ctx.triggered_id

  carla_command = f'python .\\run_experiment.py -d {chosen_dataset} -t {chosen_model} -r ar cchvae cem cem-vae clue cruds dice face_knn face_epsilon gs mace revise wachter'
  
  # open/ok/cancel button has been clicked
  if trigger == 'open':
    # open button has been clicked
    return [not is_open, carla_command]
  
  return [is_open, carla_command]

# Add Controls for the Constraint dropdown
@callback(
    Output(component_id='dataset-radar-dropdown', component_property='options'),
    Output(component_id='dataset-radar-dropdown', component_property='value'),
    Input(component_id='constraints-dropdown', component_property='value')
)
def update_constraints_dropdown(input_value):
  """
  Handles input changes to the constraints dropdown.

  Parameters
  ----------
  Constraints Dropdown : Input
    Constraints Dropdown Inputs.
         
  Returns
  -------
  DatasetRadar: List
    A list containing the options and value of the dataset radar dropdown.
  """
  dataset_list = ['adult', 'compass', 'credit']
  # if input_value == "Categorical Data":
  #   dataset_list.remove('credit')
  return [dataset_list, dataset_list[0]]

# Run the app
if __name__ == '__main__':
    app.run(debug=True)