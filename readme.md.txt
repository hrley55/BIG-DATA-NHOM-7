# Đồ án Machine Learning - Phân tích hành khách thương mại điện tử Olist (Nhóm 07)

Dự án này sử dụng Machine Learning để phân tích hành vi khách hàng, phân cụm (Clustering) và xây dựng hệ thống gợi ý sản phẩm cho nền tảng thương mại điện tử Olist (Brazil).

## 📂 Cấu trúc thư mục
* `app.py`: File chạy chính cho giao diện Web Streamlit.
* `data/`: Chứa các tệp dữ liệu sạch (.csv).
* `models/`: Chứa các mô hình máy học đã huấn luyện (.joblib, .pkl).
* `requirements.txt`: Danh sách các thư viện cần cài đặt.
* `ML_Sklearn.ipynb`: File Jupyter Notebook chi tiết quá trình huấn luyện mô hình.

## 🚀 Hướng dẫn cài đặt và chạy dự án

### 1. Tải mã nguồn
Mở Terminal/PowerShell và gõ:
```bash
git clone [https://github.com/hrley55/BIG-DATA-NHOM-7.git](https://github.com/hrley55/BIG-DATA-NHOM-7.git)
cd BIG-DATA-NHOM-7
2. Cài đặt thư viện
Đảm bảo bạn đã cài đặt Python, sau đó chạy lệnh:
pip install -r requirements.txt
3. Xem code huấn luyện mô hình
Nếu muốn xem chi tiết các bước xử lý dữ liệu và thuật toán (K-means, RFM, SVD):
Các file notebook: .ipynb
4. Khởi chạy giao diện Web (Streamlit)
Để chạy dashboard phân tích và hệ thống gợi ý:
streamlit run app.py
🛠 Công nghệ sử dụng
Ngôn ngữ: Python

Thư viện: Pandas, Scikit-learn, Streamlit, Plotly.

Mô hình: K-means Clustering, RFM Analysis, Singular Value Decomposition (SVD).

👥 Thành viên thực hiện (Nhóm 07)
Tô Nguyễn Minh Thùy (Trưởng nhóm - Xây dựng Pipeline & Web App)

Trần Khánh Ly

Nguyễn Thị Xuân Tâm

Sơn Thị Cẩm Ly

Nguyễn Hải Thùy Trâm

GVHD: Thầy Hồ Nhựt Minh
