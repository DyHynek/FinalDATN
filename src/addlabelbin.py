import os
import pandas as pd

# Đường dẫn thư mục chứa các file CSV
input_folder = "nonlinear_domain"        
output_folder = "Features_nonlinear"   
os.makedirs(output_folder, exist_ok=True)

# Duyệt tất cả file CSV
csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

for file_name in csv_files:
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name)

    df = pd.read_csv(input_path)

   
    df['Label bin'] = df['Label'].apply(lambda x: 0 if x <= 5 else 1)

    
    df.to_csv(output_path, index=False)
    print(f"✅ Đã xử lý: {file_name}")

print("Hoàn tất xử lý tất cả các file.")
