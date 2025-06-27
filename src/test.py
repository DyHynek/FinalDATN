import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu cho biểu đồ
# Tên các nghiên cứu/phương pháp
methods = [    
    "X. Zhang et al. (2021)",
    "Z. Ren et al. (2021)",
    "T. S. Delwar et al. (2025)",
    "Nghiên cứu này",
    "F. Makhmudov et al. (2024)",
    "F. Wang et al. (2025)",
]

# Độ chính xác tương ứng (ước tính từ hình ảnh)
accuracy = [ 
    62.84, # Zhang et al. (2021)
    92.71, # Ren et al. (2021)
    92.75, # Delwar et al. (2025)
    95.02,  # Nghiên cứu này
    96.54, # Makmudowar et al. (2024)
    97.93 # Wang et al. (2025)
]

# Nhãn cho từng thanh (phần bên phải)
method_labels = [
    "Dựa trên hành vi, MOL - TCE",
    "EEG, RBF-TLLH",
    "Hình ảnh/video, CNN",
    "PPG, RF",
    "Biểu cảm khuôn mặt, CNN",
    "ECG, Mạng tán xạ Wavelet"
]

# Đảo ngược thứ tự để "Nghiên cứu này" nằm trên cùng như trong hình ảnh
# methods.reverse()
# accuracy.reverse()
# method_labels.reverse()

# Tạo một mảng numpy cho vị trí các thanh
y_pos = np.arange(len(methods))

# Tạo biểu đồ
fig, ax = plt.subplots(figsize=(10, 7)) # Kích thước biểu đồ

# Lựa chọn màu sắc. Chúng ta sẽ tạo một gradient màu xanh lam/xám đậm dần
# Bạn có thể điều chỉnh các giá trị R, G, B để có màu sắc mong muốn
colors = [
    (0.2, 0.3, 0.5),  # Darkest blue-grey for Wang et al. (2025)
    (0.25, 0.35, 0.55), # Slightly lighter
    (0.3, 0.4, 0.6),  # Lighter
    (0.4, 0.5, 0.65),  # Lighter blue
    (0.5, 0.6, 0.7),  # Lighter blue-grey
    (0.6, 0.7, 0.75)   # Lightest blue for "Nghiên cứu này"
]
# Đảo ngược màu để phù hợp với thứ tự đảo ngược của dữ liệu
colors.reverse()

ax.barh(y_pos, accuracy, color=colors)

# Thêm tiêu đề và nhãn
ax.set_xlabel('Độ chính xác (%)', fontsize=12)
ax.set_title('Biểu đồ 1: So sánh độ chính xác và phương pháp (nhận trực tiếp)', fontsize=14)
ax.set_yticks(y_pos)
ax.set_yticklabels(methods, fontsize=11)

# Đặt giới hạn x để giống với biểu đồ gốc (từ 0 đến 100)
ax.set_xlim(0, 100)

# Thêm lưới dọc (dashed grid lines)
ax.xaxis.grid(True, linestyle='--', alpha=0.7)
ax.yaxis.grid(True, linestyle='--', alpha=0.0) # Tắt lưới ngang để chỉ có lưới dọc

# Đặt nhãn cho các thanh bên phải
for i, (acc, label) in enumerate(zip(accuracy, method_labels)):
    # Vị trí văn bản: ngay sau cuối thanh
    ax.text(acc + 1, y_pos[i], label, va='center', ha='left', fontsize=11)

for i, acc in enumerate(accuracy):
    # Đặt văn bản ngay trên thanh, thụt vào một chút từ cuối
    # `f'{acc}%'` định dạng số thành chuỗi và thêm ký hiệu %
    ax.text(acc - 3, y_pos[i], f'{acc}%', va='center', ha='right', color='white', fontsize=10, fontweight='bold')

# Đảo ngược trục y để thanh đầu tiên (Nghiên cứu này) ở trên cùng
ax.invert_yaxis()

# Loại bỏ khung biểu đồ bên phải và trên cùng
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Hiển thị biểu đồ
plt.tight_layout() # Đảm bảo tất cả các thành phần vừa vặn trong hình
plt.show()