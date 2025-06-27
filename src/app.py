import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash.dash_table as dt
import pandas as pd
import os
import plotly.graph_objs as go
import webbrowser
from threading import Timer
import requests



# Tên file CSV
CSV_FILE_NAME = "data.csv"
TIME_FEATURE_FILE_NAME = "heart_rate_results.csv"
WAVELET_FEATURE_FILE_NAME = "wavelet_results.csv"
NONLINEAR_FEATURE_FILE_NAME = "nonlinear_results.csv"
FREQUENCY_FEATURE_FILE_NAME = "fft_results.csv"
OUTPUT_FILE = "final_prediction.csv"

# Khởi tạo Dash app
app = dash.Dash(__name__)
app.title = "Real-Time Drowsiness Prediction"

# CSS cho bảng
table_style = {
    'style_table': {'overflowX': 'auto', 'border': '1px solid #ccc'},
    'style_header': {
        'backgroundColor': '#003366',
        'color': 'white',
        'fontWeight': 'bold',
        'textAlign': 'center'
    },
    'style_cell': {
        'textAlign': 'center',
        'padding': '5px',
        'minWidth': '80px',
        'maxWidth': '200px',
        'whiteSpace': 'normal'
    },
    'style_data_conditional': [
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': '#f2f2f2'
        }
    ]
}

# Giao diện
app.layout = html.Div([
    html.H1("Real-Time Drowsiness Prediction", style={"textAlign": "center"}),

    dcc.Graph(id='live-chart'),
    html.Div(id='status-message', style={"textAlign": "center", "fontSize": "30px", "color": "red"}),

    html.Div("🔄 Data is being updated...", style={"textAlign": "center", "marginTop": "10px"}),
    
    html.Div(id='prediction-status', style={"textAlign": "center", "fontSize": "28px", "color": "blue", "marginTop": "20px"}),

    html.H3("Prediction", style={"marginTop": "30px"}),
    dt.DataTable(id='prediction-table', page_size=5, **table_style),

    html.H3("Time Domain Features", style={"marginTop": "30px"}),
    dt.DataTable(id='time-feature-table', page_size=5, **table_style),

    html.H3("Fourier Features"),
    dt.DataTable(id='fourier-feature-table', page_size=5, **table_style),

    html.H3("Wavelet Features"),
    dt.DataTable(id='wavelet-feature-table', page_size=5, **table_style),

    html.H3("Nonlinear Features"),
    dt.DataTable(id='nonlinear-feature-table', page_size=5, **table_style),

    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
])
# Hàm gửi thông báo đến Telegram
def send_telegram_alert(message):
    token = "8126423833:AAECzoVrIg8sJLktocGEMAX8OfoH79oNHHU"     
    chat_id = "7563766946"      
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print(f"Telegram error: {e}")

# Hàm đọc CSV
def safe_read_csv(filename):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        if not df.empty:
            return df
    return pd.DataFrame()

# Callback cập nhật toàn bộ giao diện
@app.callback(
    [
        Output('live-chart', 'figure'),
        Output('status-message', 'children'),
        Output('prediction-status', 'children'),
        Output('prediction-table', 'data'),
        Output('prediction-table', 'columns'),
        Output('time-feature-table', 'data'),
        Output('time-feature-table', 'columns'),
        Output('fourier-feature-table', 'data'),
        Output('fourier-feature-table', 'columns'),
        Output('wavelet-feature-table', 'data'),
        Output('wavelet-feature-table', 'columns'),
        Output('nonlinear-feature-table', 'data'),
        Output('nonlinear-feature-table', 'columns'),
    ],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    df = safe_read_csv(CSV_FILE_NAME)

    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Scatter(
            x=df['Time (s)'],
            y=df['IR Value filtered'],
            mode='lines',
            name='IR Filtered',
            line=dict(color='black'),
            
            
        ))
        fig.update_layout(
            title='PPG signal (Real-Time)',
            xaxis_title='Time (s)',
            yaxis_title='PPG signal',
            title_font=dict(size=24),
            xaxis=dict(title_font=dict(size=18)),
            yaxis=dict(title_font=dict(size=18)),
            plot_bgcolor='white',     
            paper_bgcolor='white'
        )
        latest_ir_raw = df['IR Value raw'].iloc[-1]
        status = "⚠️ Please place your finger on the sensor." if latest_ir_raw < 50000 else ""
    else:
        fig = {}
        status = "No data available."

    # Đọc đặc trưng
    time_df = safe_read_csv(TIME_FEATURE_FILE_NAME)
    fourier_df = safe_read_csv(FREQUENCY_FEATURE_FILE_NAME)
    wavelet_df = safe_read_csv(WAVELET_FEATURE_FILE_NAME)
    nonlinear_df = safe_read_csv(NONLINEAR_FEATURE_FILE_NAME)
    prediction_df = safe_read_csv(OUTPUT_FILE)

    prediction_status = ""

    if not prediction_df.empty and 'Final Prediction' in prediction_df.columns:
        latest_pred = prediction_df['Final Prediction'].iloc[-1]
        if latest_ir_raw < 50000:
            prediction_status = "⚠️ Not detected! Please place your finger on the sensor."
        else:
            if latest_pred == 0:
                prediction_status = "🟢 Status: Alert"
            elif latest_pred == 1:
                prediction_status = "🔴 Status: Drowsy"
                send_telegram_alert("🔴 Drowsiness detected! Please take a rest.")

    def df_to_table(df):
        return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]

    return (
        fig, status,prediction_status,
        *df_to_table(prediction_df),
        *df_to_table(time_df),
        *df_to_table(fourier_df),
        *df_to_table(wavelet_df),
        *df_to_table(nonlinear_df)
    )

# Chạy ứng dụng
if __name__ == '__main__':
   
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        Timer(0.5, lambda: webbrowser.open("http://127.0.0.1:8050/")).start()

    app.run(debug=True)
