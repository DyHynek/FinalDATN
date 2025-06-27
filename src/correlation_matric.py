import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Thư mục chứa các file CSV
input_folder = "Features_wavelet"  

# Duyệt qua tất cả các file CSV
csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

# Nối tất cả các file CSV lại thành một DataFrame duy nhất
all_data = pd.DataFrame()
for file in csv_files:
    file_path = os.path.join(input_folder, file)
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=True)
    
    # Chỉ giữ lại các cột số
    df_numeric = df.select_dtypes(include='number')
    df_numeric = df_numeric.drop(columns=['Label', 'Label bin', 'Start Time (s)', 'End Time (s)'], errors='ignore')  # Nếu có cột nhãn

    all_data = pd.concat([all_data, df_numeric], ignore_index=True)

# Tính toán ma trận tương quan Pearson
corr_matrix = all_data.corr()

# Vẽ biểu đồ heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Ma trận tương quan giữa các đặc trưng từ biến đổi Wavelet")
plt.tight_layout()
plt.show()
