import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Nhập các giá trị TP, FN, FP, TN
TP = int(input("True Positive (TP): "))
FN = int(input("False Negative (FN): "))
FP = int(input("False Positive (FP): "))
TN = int(input("True Negative (TN): "))

# Tạo confusion matrix
conf_matrix = np.array([[TP, FN],
                        [FP, TN]])

# Tạo DataFrame cho dễ nhìn
df_cm = pd.DataFrame(conf_matrix, index=['Alert', 'Drowsy'],
                     columns=['Alert', 'Drowsy'])

# Vẽ bằng Seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix for Testing")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()
