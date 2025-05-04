import os
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output

app = Dash(__name__)
app.title = "NS2 TCP Performance Dashboard"

DATA_DIR = "."
protocols = ["reno", "tahoe", "vegas"]
metrics = ["throughput", "latency", "drop", "cwnd"]

def read_xg_file(filepath):
    try:
        df = pd.read_csv(filepath, sep=" ", names=["Time", "Value"])
        df = df[(df["Time"] != 0) | (df["Value"] != 0)]  # Skip the first dummy point
        return df
    except Exception:
        return pd.DataFrame(columns=["Time", "Value"])


def generate_graph(df, metric, protocol):
    return {
        "data": [go.Scatter(x=df["Time"], y=df["Value"], mode="lines", name=metric)],
        "layout": go.Layout(
            title=f"{metric.capitalize()} for {protocol.capitalize()}",
            xaxis={"title": "Time"},
            yaxis={"title": metric.capitalize()},
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f4f4f4",
            margin={"l": 40, "r": 20, "t": 30, "b": 40}
        )
    }

def calculate_summary(df):
    avg = df["Value"].mean() if not df.empty else 0
    max_val = df["Value"].max() if not df.empty else 0
    return round(avg, 2), round(max_val, 2)

app.layout = html.Div([
    html.H1("📡 NS2 TCP Protocol Analyzer", style={
        "textAlign": "center",
        "color": "#2c3e50",
        "fontFamily": "Segoe UI",
        "marginBottom": "20px"
    }),

    html.Div([
        html.Label("Select TCP Protocol:", style={"fontWeight": "bold", "fontSize": "16px"}),
        dcc.Dropdown(
            id="protocol-dropdown",
            options=[{"label": proto.capitalize(), "value": proto} for proto in protocols],
            value="reno",
            style={"width": "300px", "margin": "0 auto"}
        )
    ], style={"textAlign": "center", "marginBottom": "30px"}),

    html.Div([
        html.H3("📋 Protocol Metric Summary", style={"color": "#34495e", "fontWeight": "bold"}),
        dash_table.DataTable(
            id="summary-table",
            columns=[
                {"name": "Metric", "id": "Metric"},
                {"name": "Average", "id": "Average"},
                {"name": "Maximum", "id": "Maximum"},
            ],
            style_data={
                "backgroundColor": "#f9f9f9",
                "border": "1px solid #dcdcdc",
                "textAlign": "center",
                "fontFamily": "Segoe UI"
            },
            style_header={
                "backgroundColor": "#2c3e50",
                "color": "white",
                "fontWeight": "bold"
            },
            style_table={"overflowX": "auto", "marginBottom": "30px"},
        )
    ], style={"margin": "0 auto", "width": "90%"}),

    html.Div([
        html.Div([
            html.H4(metric.capitalize(), style={"textAlign": "center", "color": "#2980b9"}),
            dcc.Graph(id=f"{metric}-graph", config={"displayModeBar": False})
        ], style={"width": "48%", "display": "inline-block", "padding": "10px"})
        for metric in metrics
    ]),

    html.Div([
        html.H3("📊 Protocol Comparison (All TCP Variants)", style={"textAlign": "center", "color": "#2c3e50", "marginTop": "40px"}),
        html.Div([
            html.Div([
                html.H4(metric.capitalize(), style={"textAlign": "center", "color": "#8e44ad"}),
                dcc.Graph(id=f"{metric}-comparison-graph", config={"displayModeBar": False})
            ], style={"width": "48%", "display": "inline-block", "padding": "10px"})
            for metric in metrics
        ])
    ])
])

@app.callback(
    [Output(f"{metric}-graph", "figure") for metric in metrics] +
    [Output("summary-table", "data")] +
    [Output(f"{metric}-comparison-graph", "figure") for metric in metrics],
    [Input("protocol-dropdown", "value")]
)
def update_protocol_data(selected_proto):
    figures = []
    summary_data = []

    # For selected protocol
    for metric in metrics:
        file_path = os.path.join(DATA_DIR, f"{metric}-{selected_proto}.xg")
        df = read_xg_file(file_path)
        figures.append(generate_graph(df, metric, selected_proto))
        avg, max_val = calculate_summary(df)
        summary_data.append({"Metric": metric.capitalize(), "Average": avg, "Maximum": max_val})

    # For comparison of all protocols
    comparison_figures = []
    for metric in metrics:
        traces = []
        for proto in protocols:
            file_path = os.path.join(DATA_DIR, f"{metric}-{proto}.xg")
            df = read_xg_file(file_path)
            traces.append(go.Scatter(x=df["Time"], y=df["Value"], mode="lines", name=proto.capitalize()))
        layout = go.Layout(
            title=f"{metric.capitalize()} Comparison (All Protocols)",
            xaxis={"title": "Time"},
            yaxis={"title": metric.capitalize()},
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f4f4f4",
            margin={"l": 40, "r": 20, "t": 30, "b": 40}
        )
        comparison_figures.append({"data": traces, "layout": layout})

    return figures + [summary_data] + comparison_figures

if __name__ == '__main__':
    print("✨ Running TCP Protocol Dashboard")
    app.run(debug=True)
