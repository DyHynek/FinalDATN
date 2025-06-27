# import os
# import pandas as pd
# import numpy as np
# import pywt
# from scipy.stats import kurtosis

# # 🔹 Thư mục chứa các file CSV
# input_folder = "predict_data"
# output_folder = "Wavelet_Transform"
# os.makedirs(output_folder, exist_ok=True)

# # 🔹 Cấu hình cửa sổ trượt
# window_size = 60  # giây
# step_size = 1     # giây

# # 🔹 Danh sách tất cả file CSV
# csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

# for file_name in csv_files:
#     file_path = os.path.join(input_folder, file_name)
#     df = pd.read_csv(file_path)

#     if "IR Value filtered" in df.columns and "Time (s)" in df.columns and "Label_Detail" in df.columns:
#         df = df.drop_duplicates(subset=["Time (s)"]).sort_values(by="Time (s)")
#         time = np.array(df["Time (s)"])
#         ir_signal = np.array(df["IR Value filtered"])
#         label = np.array(df["Label_Detail"])

#         start_time = time[0]
#         end_time = time[-1]
#         result = []

#         for current_start in np.arange(start_time, end_time - window_size, step_size):
#             current_end = current_start + window_size
#             mask = (time >= current_start) & (time <= current_end)
#             time_window = time[mask]
#             ir_window = ir_signal[mask]
#             label_window = label[mask]

#             if len(ir_window) < 2:
#                 continue

#             dominant_label = pd.Series(label_window).mode()[0] if len(label_window) > 0 else "unknown"

#             try:
#                 coeffs = pywt.wavedec(ir_window, wavelet='coif5', level=4)
#                 A4, D4, D3, D2, D1 = coeffs  # Lấy đúng thứ tự D1-D4-A4
#                 kurt_D1 = kurtosis(D1)
#                 kurt_D2 = kurtosis(D2)
#                 kurt_D3 = kurtosis(D3)
#                 kurt_A4 = kurtosis(A4)
#             except Exception as e:
#                 print(f"⚠️ Lỗi wavelet ở cửa sổ [{current_start:.1f}s - {current_end:.1f}s] trong file {file_name}: {e}")
#                 continue

#             result.append([
#                 current_start, current_end,
#                 kurt_D1, kurt_D2, kurt_D3, kurt_A4,
#                 dominant_label
#             ])

#         # 🔹 Lưu kết quả
#         output_filename = f"Wavelet_result_{os.path.splitext(file_name)[0]}.csv"
#         output_path = os.path.join(output_folder, output_filename)

#         result_df = pd.DataFrame(result, columns=[
#             "Start Time (s)", "End Time (s)",
#             "Kurtosis D1", "Kurtosis D2", "Kurtosis D3", "Kurtosis A4",
#             "Label"
#         ])
#         result_df.to_csv(output_path, index=False)
#         print(f"✅ Đã xử lý và lưu: {output_filename}")
#     else:
#         print(f"⚠️ File {file_name} thiếu cột cần thiết.")
import pandas as pd
import numpy as np
import pywt
from scipy.stats import kurtosis

# Đọc file CSV
df = pd.read_csv("data/data_Cong_11062025.csv")

if "IR Value filtered" in df.columns and "Time (s)" in df.columns and "Label" in df.columns:
    df = df.drop_duplicates(subset=["Time (s)"]).sort_values(by="Time (s)")
    time = np.array(df["Time (s)"])
    ir_signal = np.array(df["IR Value filtered"])
    label = np.array(df["Label"])

    window_size = 60  # giây
    step_size = 1     # giây
    start_time = time[0]
    end_time = time[-1]
    result = []

    for current_start in np.arange(start_time, end_time - window_size, step_size):
        current_end = current_start + window_size
        mask = (time >= current_start) & (time <= current_end)
        time_window = time[mask]
        ir_window = ir_signal[mask]
        label_window = label[mask]

        if len(ir_window) < 2:
            continue

        # Gán nhãn chiếm ưu thế
        dominant_label = pd.Series(label_window).mode()[0] if len(label_window) > 0 else "unknown"

        # Biến đổi wavelet mức 4 với coif5
        coeffs = pywt.wavedec(ir_window, wavelet='coif5', level=4)
        A4, D4, D3, D2, D1 = coeffs  # Giải nén hệ số (ngược lại với thứ tự trả về)

        # Tính kurtosis
        kurt_D1 = kurtosis(D1)
        kurt_D2 = kurtosis(D2)
        kurt_D3 = kurtosis(D3)
        kurt_A4 = kurtosis(A4)

        result.append([
            current_start, current_end, kurt_D1, kurt_D2, kurt_D3, kurt_A4, dominant_label
        ])

    # Lưu kết quả
    result_df = pd.DataFrame(result, columns=[
        "Start Time (s)", "End Time (s)",
        "Kurtosis D1", "Kurtosis D2", "Kurtosis D3", "Kurtosis A4", "Label"
    ])
    result_df.to_csv("wavelet_results.csv", index=False)
    print("✅ Đã lưu các đặc trưng vào 'wavelet_results.csv'")
else:
    print("⚠️ Không tìm thấy cột cần thiết trong file CSV.")