import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Tên các thư mục chứa dữ liệu (giữ nguyên như mã gốc của bạn)
time_domain_dir = 'Features_time'
nonlinear_domain_dir = 'Features_nonlinear'
wavelet_dir = 'Features_wavelet'
frequence_dir = 'Features_fourier'

# Hàm để đọc tất cả các file CSV từ một thư mục và kết hợp chúng thành một DataFrame
def load_and_combine_all(directory, prefix, drop_cols):
    all_files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.csv')]
    df_list = []
    for file in all_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=True)
        df_dropped = df.drop(columns=drop_cols, errors='ignore')
        df_list.append(df_dropped)
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df
    return None

# Đọc và kết hợp dữ liệu cho từng miền 
df_thoi_gian_all = load_and_combine_all(time_domain_dir, 'time_data', ['Start Time (s)', 'End Time (s)', 'Label'])
df_phi_tuyen_all = load_and_combine_all(nonlinear_domain_dir, 'Nonlinear_result_data', ['Time start', 'Time end', 'Label'])
df_wavelet_all = load_and_combine_all(wavelet_dir, 'Wavelet_result_data', ['Start Time (s)', 'End Time (s)', 'Label'])
df_tanso_all = load_and_combine_all(frequence_dir, 'FFT_result_data', ['Start Time (s)', 'End Time (s)', 'Label'])

# Hàm để huấn luyện và đánh giá mô hình KNN trên một DataFrame
def train_evaluate_knn_domain(df, domain_name, model_filename, n_neighbors=5):
    if df is not None:
        X = df.drop('Label bin', axis=1, errors='ignore')
        y = df['Label bin']

        # Đảm bảo rằng có đủ mẫu cho cả tập huấn luyện và tập kiểm tra
        if len(y.unique()) < 2:
            print(f"\nKhông đủ lớp để huấn luyện mô hình cho {domain_name}. Cần ít nhất 2 lớp.")
            return None
        if len(X) < 2: # Ít nhất 2 mẫu để tách
            print(f"\nKhông đủ mẫu để huấn luyện mô hình cho {domain_name}.")
            return None

        # Tách dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Khởi tạo và huấn luyện mô hình K-Nearest Neighbors
        model = KNeighborsClassifier(n_neighbors=n_neighbors) # Sử dụng n_neighbors được truyền vào
        model.fit(X_train, y_train)

        # Dự đoán trên tập kiểm tra
        y_pred = model.predict(X_test)

        # Đánh giá mô hình
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nĐánh giá mô hình KNN cho {domain_name}:")
        print("Độ chính xác:", accuracy)
        print("Báo cáo phân loại:\n", classification_report(y_test, y_pred))
        print("Ma trận nhầm lẫn:\n", confusion_matrix(y_test, y_pred))

        # Lưu mô hình
        joblib.dump(model, model_filename)
        print(f"✅ Đã lưu mô hình KNN {domain_name} vào '{model_filename}'")
        return accuracy
    else:
        print(f"\nKhông có dữ liệu cho {domain_name}.")
        return None

# Huấn luyện và đánh giá cho từng miền sử dụng KNN
accuracy_thoi_gian_knn = train_evaluate_knn_domain(df_thoi_gian_all, "miền thời gian", "model_time_domain_knn.pkl", n_neighbors=20)
accuracy_phi_tuyen_knn = train_evaluate_knn_domain(df_phi_tuyen_all, "đặc trưng phi tuyến", "model_nonlinear_knn.pkl", n_neighbors=20) # Ví dụ: thay đổi n_neighbors
accuracy_wavelet_knn = train_evaluate_knn_domain(df_wavelet_all, "biến đổi wavelet", "model_wavelet_knn.pkl", n_neighbors=20)
accuracy_tanso_knn = train_evaluate_knn_domain(df_tanso_all, "biến đổi Fourier", "model_frequency_knn.pkl", n_neighbors=20)

# In ra độ chính xác của từng miền (KNN)
print("\nĐộ chính xác của từng miền (KNN):")
if accuracy_thoi_gian_knn is not None:
    print(f"Miền thời gian (KNN): {accuracy_thoi_gian_knn:.4f}")
if accuracy_phi_tuyen_knn is not None:
    print(f"Miền phi tuyến (KNN): {accuracy_phi_tuyen_knn:.4f}")
if accuracy_wavelet_knn is not None:
    print(f"Biến đổi Wavelet (KNN): {accuracy_wavelet_knn:.4f}")
if accuracy_tanso_knn is not None:
    print(f"Biến đổi Fourier (KNN): {accuracy_tanso_knn:.4f}")