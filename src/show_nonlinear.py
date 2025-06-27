import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib import colormaps 
from matplotlib.patches import Patch
# 🔹 1. Đọc dữ liệu
file_path = "Features_nonlinear/Nonlinear_result_data_Cong_12052025.csv"
df = pd.read_csv(file_path)

metrics = [
    "AMI",
    "FNN",
    "Lyapunov",
    "ARE",
    "DFA"
]

time = df["Time start"]
labels = df["Label"]

# Tìm các đoạn liên tiếp có cùng label
label_segments = []
current_label = labels.iloc[0]
start_time = time.iloc[0]

for i in range(1, len(labels)):
    if labels.iloc[i] != current_label:
        end_time = time.iloc[i]
        label_segments.append((start_time, end_time, current_label))
        current_label = labels.iloc[i]
        start_time = time.iloc[i]
label_segments.append((start_time, time.iloc[-1], current_label))

# Màu cho 9 label
unique_labels = list(range(1, 10))
cmap = colormaps["tab10"]
label_color_map = {label: cmap(i / 9) for i, label in enumerate(range(1, 10))}

# Vẽ biểu đồ
plt.figure(figsize=(12, 8))

for i, metric in enumerate(metrics):
    plt.subplot(len(metrics), 1, i + 1)

    y = df[metric]
    y_smooth = savgol_filter(y, window_length=5, polyorder=2) if len(y) >= 5 else y

    plt.plot(time, y_smooth, label=f"{metric} (smoothed)", color="blue", linewidth=1.5)
    plt.scatter(time, y, color="lightgray", label="Raw", s=10)

    for start, end, label in label_segments:
        color = label_color_map.get(label, "gray")
        plt.axvspan(start, end, alpha=0.2, color=color)

    plt.title(metric)
    plt.xlabel("Time (s)")
    plt.ylabel(metric)
    plt.grid(True)

    if i == 0:
        legend_elements = [
            Patch(facecolor=label_color_map[label], alpha=0.3, label=f'Label {label}')
            for label in unique_labels
        ]
        plt.legend(handles=legend_elements, loc='upper right', ncol=3)

plt.tight_layout()
plt.show()