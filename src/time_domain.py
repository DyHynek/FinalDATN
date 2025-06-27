# import os
# import pandas as pd
# import numpy as np
# from scipy.signal import find_peaks

# # Thư mục chứa file CSV gốc
# input_folder = "predict_data"  

# # Thư mục lưu kết quả
# output_folder = "Time_domain"
# os.makedirs(output_folder, exist_ok=True)

# # Cấu hình cửa sổ trượt
# window_size = 60
# step_size = 1

# for file_name in os.listdir(input_folder):
#     if file_name.endswith(".csv"):
#         file_path = os.path.join(input_folder, file_name)
#         df = pd.read_csv(file_path)

#         if "Time (s)" in df.columns and "IR Value filtered" in df.columns and "Label_Detail" in df.columns:
#             df = df.drop_duplicates(subset=["Time (s)"]).sort_values(by=["Time (s)"])
#             time = np.array(df["Time (s)"])
#             ir_signal = np.array(df["IR Value filtered"])
#             labels = np.array(df["Label_Detail"])

#             start_time = time[0]
#             end_time = time[-1]

#             bpm_results = []

#             for current_start in np.arange(start_time, end_time - window_size, step_size):
#                 current_end = current_start + window_size
#                 mask = (time >= current_start) & (time <= current_end)
#                 time_window = time[mask]
#                 ir_signal_window = ir_signal[mask]

#                 if len(time_window) < 2:
#                     continue

#                 # Lấy label tại thời điểm current_end gần nhất
#                 idx_label = (np.abs(time - current_end)).argmin()
#                 label_at_end = labels[idx_label]

#                 peaks, _ = find_peaks(ir_signal_window, height=np.mean(ir_signal_window),
#                                       distance=0.5 * (1 / np.mean(np.diff(time_window))))
#                 valleys, _ = find_peaks(-ir_signal_window, height=-np.mean(ir_signal_window),
#                                         distance=0.3 * (1 / np.mean(np.diff(time_window))))

#                 rising_times = []
#                 for v in valleys:
#                     future_peaks = peaks[peaks > v]
#                     if len(future_peaks) == 0:
#                         continue
#                     p = future_peaks[0]
#                     rising_times.append(time_window[p] - time_window[v])

#                 if len(peaks) >= 2:
#                     peak_times = time_window[peaks]
#                     intervals = np.diff(peak_times)
#                     avg_interval = np.mean(intervals)
#                     std_interval = np.std(intervals)
#                 else:
#                     avg_interval = np.nan
#                     std_interval = np.nan

#                 if len(peaks) > 0 and len(valleys) > 0:
#                     avg_peak = np.mean(ir_signal_window[peaks])
#                     avg_valley = np.mean(ir_signal_window[valleys])
#                     amplitude = avg_peak - avg_valley
#                 else:
#                     amplitude = np.nan

#                 avg_rising_time = np.mean(rising_times) if rising_times else np.nan

#                 bpm = (len(peaks) / window_size) * 60

#                 bpm_results.append([
#                     current_start, current_end,
#                     bpm, avg_interval, std_interval, amplitude,
#                     avg_rising_time, label_at_end
#                 ])

#             # Lưu kết quả file này
#             output_df = pd.DataFrame(bpm_results, columns=[
#                 "Start Time (s)", "End Time (s)", "BPM",
#                 "AVG Interval (s)", "STD Interval (s)", "Amplitude",
#                 "Time V2P (s)", "Label"
#             ])
#             output_path = os.path.join(output_folder, f"time_{file_name}")
#             output_df.to_csv(output_path, index=False)
#             print(f"✅ Đã xử lý và lưu kết quả file: {output_path}")

#         else:
#             print(f"⚠️ File {file_name} không có đầy đủ cột cần thiết!")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 🔹 1. Đọc file CSV
file_path = "data/data_Cong_11062025.csv"  # Đổi thành đường dẫn file của bạn
df = pd.read_csv(file_path)

# 🔹 2. Kiểm tra dữ liệu
if "Time (s)" in df.columns and "IR Value filtered" in df.columns and "Label" in df.columns:
    df = df.drop_duplicates(subset=["Time (s)"]).sort_values(by=["Time (s)"])
    time = np.array(df["Time (s)"])
    ir_signal = np.array(df["IR Value filtered"])
    labels = np.array(df["Label"])

    # 🔹 3. Thông số cửa sổ trượt
    window_size = 60 #Kích thước cửa sổ (10 giây)
    step_size = 1# Dịch chuyển mỗi lần 1 giây

    start_time = time[0]
    end_time = time[-1]

    bpm_results = []

    # 🔹 4. Duyệt qua các cửa sổ trượt
    for current_start in np.arange(start_time, end_time - window_size, step_size):
        current_end = current_start + window_size
        mask = (time >= current_start) & (time <= current_end)
        time_window = time[mask]
        ir_signal_window = ir_signal[mask]
        label_window = labels[mask]

        # Gán nhãn phổ biến nhất trong cửa sổ
        if len(label_window) > 0:
            label_counts = pd.Series(label_window).value_counts()
            dominant_label = label_counts.idxmax()
        else:
            dominant_label = "unknown"

        # Lọc dữ liệu trong khoảng [current_start, current_end]
        mask = (time >= current_start) & (time <= current_end)
        time_window = time[mask]
        ir_signal_window = ir_signal[mask]

        if len(time_window) < 2:
            continue  # Bỏ qua nếu dữ liệu quá ít

        # 🔹 5a. Phát hiện đỉnh PPG (tương ứng với nhịp tim)
        peaks, _ = find_peaks(ir_signal_window, height=np.mean(ir_signal_window), distance=0.5 * (1 / np.mean(np.diff(time_window))) )
        # 🔹 5b. Phát hiện đáy PPG
        valleys, _ = find_peaks(-ir_signal_window, height=-np.mean(ir_signal_window), distance=0.3 * (1 / np.mean(np.diff(time_window))))
        #valleys = [i for i in valleys if ir_signal_window[i] < 0]
        rising_times = []
        for v in valleys:
            future_peaks = peaks[peaks > v]
            if len(future_peaks) == 0:
                continue
            p = future_peaks[0]  # Lấy đỉnh gần nhất sau đáy
            t_valley = time_window[v]
            t_peak = time_window[p]
            rising_times.append(t_peak - t_valley)
        # 🔹 5c. Tính khoảng thời gian trung bình giữa các đỉnh
        if len(peaks) >= 2:
            peak_times = time_window[peaks]
            intervals = np.diff(peak_times)
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
        else:
            avg_interval = np.nan  # Không đủ đỉnh để tính
            std_interval = np.nan

        if len(peaks) > 0 and len(valleys) > 0:
            avg_peak = np.mean(ir_signal_window[peaks])
            avg_valley = np.mean(ir_signal_window[valleys])
            amplitude = avg_peak - avg_valley
        else:
            amplitude = np.nan

        if rising_times:
            avg_rising_time = np.mean(rising_times)
        else:
            avg_rising_time = np.nan
        
        
        
        # Tính BPM
        num_beats = len(peaks)
        bpm = (num_beats / window_size) * 60  # Chuyển đổi sang BPM

        bpm_results.append([current_start, current_end, bpm, avg_interval, std_interval, amplitude, avg_rising_time, dominant_label])

        # # 🔹 6. Vẽ tín hiệu + nhịp tim
        # plt.figure(figsize=(10, 4))
        # plt.plot(time_window, ir_signal_window, label="IR Signal", color="b")
        # plt.scatter(time_window[peaks], ir_signal_window[peaks], color="r", marker="o", label="Detected Peaks")
        # plt.scatter(time_window[valleys], ir_signal_window[valleys], color="y", marker="o", label="Detected Valleys")
        # plt.xlabel("Time (s)")
        # plt.ylabel("IR Value filtered")
        # plt.title(f"PPG Signal & Heart Rate: {bpm:.1f} BPM ({current_start:.1f}s - {current_end:.1f}s)")
        # plt.legend()
        # plt.grid()
        # plt.show()

    # 🔹 7. Lưu BPM vào file CSV
    bpm_df = pd.DataFrame(bpm_results, columns=["Start Time (s)", "End Time (s)", "BPM", "AVG Interval (s)", "STD Interval (s)", "Amplitude", "Time V2P (s)", "Label"])
    bpm_df.to_csv("heart_rate_results.csv", index=False)

    print("✅ Đã lưu nhịp tim trung bình vào 'heart_rate_results.csv'")
else:
    print("⚠️ Cột dữ liệu không đúng! Hãy kiểm tra lại file CSV.")