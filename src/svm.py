import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
# Tên các thư mục chứa dữ liệu
time_domain_dir = 'Features_time'
nonlinear_domain_dir = 'Features_nonlinear'
wavelet_dir = 'Features_wavelet'
frequence_dir = 'Features_fourier'
# Hàm để đọc và kết hợp tất cả các file CSV
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

# Đọc dữ liệu
df_thoi_gian_all = load_and_combine_all(time_domain_dir, 'time_data', ['Start Time (s)', 'End Time (s)', 'Label'])
df_phi_tuyen_all = load_and_combine_all(nonlinear_domain_dir, 'Nonlinear_result_data', ['Time start', 'Time end', 'Label'])
df_wavelet_all = load_and_combine_all(wavelet_dir, 'Wavelet_result_data', ['Start Time (s)', 'End Time (s)', 'Label'])
df_tanso_all = load_and_combine_all(frequence_dir, 'FFT_result_data', ['Start Time (s)', 'End Time (s)', 'Label'])

def plot_confusion_matrix(cm, domain_name, labels=['Alert', 'Drowsy']):
    plt.figure(figsize=(8, 6))  
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix for {domain_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Huấn luyện và đánh giá với SVM
def train_evaluate_svm(df, domain_name, model_filename):
    if df is not None:
        X = df.drop('Label bin', axis=1, errors='ignore')
        y = df['Label bin']

        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

        # Khởi tạo và huấn luyện mô hình SVM
        model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
        model.fit(X_train, y_train)

        # Dự đoán
        y_pred = model.predict(X_test)

        # Đánh giá
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nĐánh giá mô hình SVM cho {domain_name}:")
        print("Độ chính xác:", accuracy)
        print("Báo cáo phân loại:\n", classification_report(y_test, y_pred))
        print("Ma trận nhầm lẫn:\n", confusion_matrix(y_test, y_pred))

        plot_confusion_matrix(confusion_matrix(y_test, y_pred), domain_name, labels=['Alert', 'Drowsy'])
        # Lưu mô hình và scaler
        joblib.dump({'model': model, 'scaler': scaler}, model_filename)
        print(f"✅ Đã lưu mô hình {domain_name} vào '{model_filename}'")
        return accuracy
    else:
        print(f"\nKhông có dữ liệu cho {domain_name}.")
        return None

# Huấn luyện và đánh giá cho từng miền
accuracy_svm_thoi_gian = train_evaluate_svm(df_thoi_gian_all, "Time", "svm_model_time_domain.pkl")
accuracy_svm_phi_tuyen = train_evaluate_svm(df_phi_tuyen_all, "Nonlinear", "svm_model_nonlinear.pkl")
accuracy_svm_wavelet = train_evaluate_svm(df_wavelet_all, "Wavelet", "svm_model_wavelet.pkl")
accuracy_svm_tanso = train_evaluate_svm(df_tanso_all, "Fourier", "svm_model_fourier.pkl")
# In độ chính xác
print("\nĐộ chính xác SVM của từng miền:") 
if accuracy_svm_thoi_gian is not None:
    print(f"Miền thời gian: {accuracy_svm_thoi_gian:.4f}")
if accuracy_svm_phi_tuyen is not None:
    print(f"Miền phi tuyến: {accuracy_svm_phi_tuyen:.4f}")
if accuracy_svm_wavelet is not None:
    print(f"Biến đổi Wavelet: {accuracy_svm_wavelet:.4f}")
if accuracy_svm_tanso is not None:
    print(f"Biến đổi Fourier: {accuracy_svm_tanso:.4f}")
