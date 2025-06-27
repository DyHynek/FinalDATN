# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter

# # 🔹 1. Đọc dữ liệu
# file_path = "Wavelet_Transform/wavelet_results_Cong_14052025.csv"
# df = pd.read_csv(file_path)

# metrics = [
#     "Kurtosis D1","Kurtosis D2","Kurtosis D3","Kurtosis A4"
# ]

# time = df["Start Time (s)"]
# labels = df["Label"]

# # 🔹 2. Tìm các đoạn liên tiếp có cùng label
# label_segments = []
# current_label = labels.iloc[0]
# start_time = time.iloc[0]

# for i in range(1, len(labels)):
#     if labels.iloc[i] != current_label:
#         end_time = time.iloc[i]
#         label_segments.append((start_time, end_time, current_label))
#         current_label = labels.iloc[i]
#         start_time = time.iloc[i]

# # Thêm đoạn cuối cùng
# label_segments.append((start_time, time.iloc[-1], current_label))

# # 🔹 3. Bản đồ màu cho từng label
# label_color_map = {
#     0: "red",
#     1: "green"
# }

# # 🔹 4. Vẽ biểu đồ + tô vùng nhãn
# plt.figure(figsize=(14, 8))

# for i, metric in enumerate(metrics):
#     plt.subplot(len(metrics), 1, i + 1)

#     y = df[metric]
#     y_smooth = savgol_filter(y, window_length=5, polyorder=2) if len(y) >= 5 else y

#     plt.plot(time, y_smooth, label=f"{metric} (smoothed)", color="blue")
#     plt.scatter(time, y, color="lightgray", label="Raw", s=10)

#     # 🔸 Vẽ các vùng nhãn màu
#     for start, end, label in label_segments:
#         color = label_color_map.get(label, "gray")  # Mặc định màu xám nếu không xác định
#         plt.axvspan(start, end, alpha=0.2, color=color)

#     plt.title(metric)
#     plt.xlabel("Time (s)")
#     plt.ylabel(metric)
#     plt.grid(True)

#     # Chỉ hiển thị chú thích 1 lần
#     if i == 0:
#         from matplotlib.patches import Patch
#         legend_elements = [
#             Patch(facecolor='red', alpha=0.2, label='Label 0'),
#             Patch(facecolor='green', alpha=0.2, label='Label 1')
#         ]
#         plt.legend(handles=legend_elements, loc='upper right')

# plt.tight_layout()
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.patches import Patch
from matplotlib import colormaps 

# Đọc dữ liệu
file_path = "Wavelet_Transform/Wavelet_result_data_04042025.csv"
df = pd.read_csv(file_path)

metrics = [
    "Kurtosis D1","Kurtosis D2","Kurtosis D3","Kurtosis A4"
]

time = df["Start Time (s)"]
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
