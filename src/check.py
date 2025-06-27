import pandas as pd
import os

# Tên các thư mục chứa dữ liệu
time_domain_dir = 'Features_time'
nonlinear_domain_dir = 'Features_nonlinear'
wavelet_dir = 'Features_wavelet'
frequence_dir = 'Features_fourier'

# Hàm đọc và kết hợp dữ liệu từ nhiều file trong thư mục
def load_and_combine_all(directory, prefix, drop_cols):
    all_files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.csv')]
    df_list = []
    for file in all_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=True) 
        df_dropped = df.drop(columns=drop_cols, errors='ignore')
        df_dropped['source_file'] = file  
        df_list.append(df_dropped)
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df
    return None

# Đọc dữ liệu từ các miền
df_thoi_gian_all = load_and_combine_all(time_domain_dir, 'time_data', ['Start Time (s)', 'End Time (s)', 'Label'])
df_phi_tuyen_all = load_and_combine_all(nonlinear_domain_dir, 'Nonlinear_result_data', ['Time start', 'Time end', 'Label'])
df_phi_tuyen_all.to_csv("b.csv",index=False)
df_wavelet_all = load_and_combine_all(wavelet_dir, 'Wavelet_result_data', ['Start Time (s)', 'End Time (s)', 'Label'])
df_wavelet_all.to_csv("a.csv",index=False)
df_tanso_all = load_and_combine_all(frequence_dir, 'FFT_result_data', ['Start Time (s)', 'End Time (s)', 'Label'])


# Hàm kiểm tra giá trị NaN trong DataFrame
def check_for_nan_values(df, df_name):
    if df is None:
        print(f"\nDataFrame '{df_name}' không có dữ liệu.")
        return

    print(f"\n--- Kiểm tra giá trị NaN cho DataFrame: {df_name} ---")

    # 1. Tổng số NaN
    total_nan = df.isnull().sum().sum()
    if total_nan == 0:
        print(f"✅ DataFrame '{df_name}' không chứa bất kỳ giá trị NaN nào.")
    else:
        print(f"❌ DataFrame '{df_name}' chứa TỔNG CỘNG {total_nan} giá trị NaN.")

    # 2. NaN theo từng cột
    nan_by_column = df.isnull().sum()
    columns_with_nan = nan_by_column[nan_by_column > 0]
    if not columns_with_nan.empty:
        print("\nSố lượng giá trị NaN theo từng cột:")
        print(columns_with_nan)
    else:
        print("\nKhông có cột nào chứa giá trị NaN.")

    # 3. Phần trăm NaN theo cột
    total_rows = len(df)
    if total_rows > 0:
        percent_nan_by_column = (nan_by_column / total_rows) * 100
        columns_with_nan_percent = percent_nan_by_column[percent_nan_by_column > 0]
        if not columns_with_nan_percent.empty:
            print("\nPhần trăm giá trị NaN theo từng cột:")
            print(columns_with_nan_percent.apply(lambda x: f"{x:.2f}%"))
    else:
        print("\nDataFrame rỗng, không thể tính phần trăm NaN.")

    # 4. Các hàng chứa NaN
    rows_with_nan = df[df.isnull().any(axis=1)]
    if not rows_with_nan.empty:
        print(f"\nCó {len(rows_with_nan)} hàng chứa ít nhất một giá trị NaN.")
        print("Một số hàng đầu tiên chứa NaN:")
        print(rows_with_nan.head())
    else:
        print("\nKhông có hàng nào chứa giá trị NaN.")

    # 5. Tên file chứa NaN 
    if 'source_file' in df.columns:
        files_with_nan = rows_with_nan['source_file'].unique()
        if len(files_with_nan) > 0:
            print(f"\n📂 Các file chứa giá trị NaN:")
            for f in files_with_nan:
                print(f"- {f}")

# Kiểm tra NaN cho từng miền
check_for_nan_values(df_thoi_gian_all, "df_thoi_gian_all (Miền thời gian)")
check_for_nan_values(df_phi_tuyen_all, "df_phi_tuyen_all (Đặc trưng phi tuyến)")
check_for_nan_values(df_wavelet_all, "df_wavelet_all (Biến đổi Wavelet)")
check_for_nan_values(df_tanso_all, "df_tanso_all (Biến đổi Fourier)")
