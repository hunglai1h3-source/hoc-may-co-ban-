# -*- coding: utf-8 -*-
# ITA202 - Lab 1: Tổng quan Học máy và cài đặt môi trường
# Chạy: python ITA202_Lab1.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.linear_model import LinearRegression


# BÀI 1: LÀM QUEN VỚI NUMPY VÀ PANDAS
print("\nBÀI 1: NUMPY VÀ PANDAS")

# np.arange không lấy cận cuối, nên dùng 11 để có đủ các số từ 1 đến 10.
mang = np.arange(1, 11)
print("Mảng NumPy:", mang)

# Dữ liệu minh họa của đúng 5 sinh viên, gồm 3 cột theo yêu cầu.
sinh_vien = {
    "Họ tên": [
        "Nguyễn Văn An", "Trần Thị Bình", "Lê Minh Châu",
        "Phạm Quốc Dũng", "Hoàng Thu Hà"
    ],
    "Tuổi": [18, 19, 18, 20, 19],
    "Điểm toán": [8.0, 7.5, 9.0, 6.5, 8.5]
}
df = pd.DataFrame(sinh_vien)
print("\nBảng thông tin sinh viên:")
print(df)

print("\nThông tin tổng quan của DataFrame:")
df.info()

print("\nThống kê các cột số:")
print(df.describe())

diem_trung_binh = df["Điểm toán"].mean()
print(f"\nĐiểm trung bình môn toán: {diem_trung_binh:.2f}")


# BÀI 2: PHÂN LOẠI BÀI TOÁN MACHINE LEARNING
# 1. Dự đoán giá nhà dựa trên diện tích:
#    Supervised (có giám sát) - Hồi quy.
#    Vì dữ liệu huấn luyện có diện tích và giá nhà thực tế làm nhãn.
# 2. Gom nhóm khách hàng dựa trên hành vi mua sắm:
#    Unsupervised (không giám sát) - Phân cụm.
#    Vì chưa có nhãn nhóm, mô hình tự tìm các khách hàng tương đồng.
# 3. Phân loại email là Spam hay không:
#    Supervised (có giám sát) - Phân loại nhị phân.
#    Vì dữ liệu huấn luyện có nhãn Spam hoặc không Spam.
# 4. Nhận diện chữ viết tay từ hình ảnh:
#    Supervised (có giám sát) - Phân loại nhiều lớp.
#    Với cách làm thông thường, ảnh huấn luyện có nhãn ký tự/chữ số đúng.

print("\nBÀI 2: PHÂN LOẠI BÀI TOÁN MACHINE LEARNING")
print("1. Giá nhà: Supervised - Hồi quy.")
print("2. Nhóm khách hàng: Unsupervised - Phân cụm.")
print("3. Email Spam: Supervised - Phân loại nhị phân.")
print("4. Chữ viết tay: Supervised - Phân loại nhiều lớp.")


# BÀI 3: CHẠY THỬ MÔ HÌNH VỚI SCIKIT-LEARN
print("\nBÀI 3: MÔ HÌNH ĐƠN GIẢN TRÊN TẬP IRIS")

# Bước 1: Load dữ liệu. X là đặc trưng, y là nhãn loài hoa (0, 1, 2).
iris = load_iris()
X = iris.data
y = iris.target
print("Kích thước X:", X.shape)
print("Kích thước y:", y.shape)
print("Các loài hoa:", iris.target_names)

# Bước 2: Khởi tạo model theo gợi ý của đề.
model = LinearRegression()

# Bước 3: Fit model (huấn luyện).
model.fit(X, y)

# Bước 4: Predict trên 3 mẫu, mỗi mẫu thuộc một loài hoa khác nhau.
chi_so_mau = [0, 50, 100]
du_doan = model.predict(X[chi_so_mau])
ket_qua = pd.DataFrame({
    "Chỉ số mẫu": chi_so_mau,
    "Nhãn thật": y[chi_so_mau],
    "Dự đoán hồi quy": du_doan
})
print("\nKết quả dự đoán:")
print(ket_qua.round(4))

# Iris là bài toán phân loại. LinearRegression trả về số liên tục,
# không phải nhãn lớp hay xác suất. Ở đây chỉ minh họa Fit/Predict.
# Các mẫu dự đoán nằm trong tập đã học, không dùng để đánh giá tổng quát hóa.
print("\nLưu ý: Dự đoán hồi quy có thể là số thập phân, không phải nhãn lớp.")
print("Đây là minh họa trên dữ liệu đã học, chưa phải đánh giá trên dữ liệu mới.")


# BÀI 4: THU THẬP VÀ KIỂM TRA DỮ LIỆU
print("\nBÀI 4: TẢI VÀ KIỂM TRA TẬP DIGITS")

digits = load_digits()
X_digits = digits.data
y_digits = digits.target

print("Hình dạng dữ liệu:", X_digits.shape)
print("Hình dạng nhãn:", y_digits.shape)
print("Hình dạng tập ảnh:", digits.images.shape)
print("Nhãn của mẫu đầu tiên:", y_digits[0])

# Mỗi ảnh có 8 x 8 pixel; một hàng dữ liệu chứa 64 giá trị pixel.
plt.figure(figsize=(4, 4))
plt.imshow(digits.images[0], cmap="gray", interpolation="nearest")
plt.title(f"Mẫu đầu tiên - Nhãn thật: {y_digits[0]}")
plt.axis("off")
plt.tight_layout()
plt.show()
