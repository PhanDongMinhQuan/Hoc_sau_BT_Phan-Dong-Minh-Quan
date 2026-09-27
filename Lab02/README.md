# Lab02: Qui Trình Máy Học Cho Dự Đoán Giá Nhà (House Prices Advanced Regression)
**Học phần**: Học sâu (Deep Learning) - **Mã học phần**: 858048  
**Lớp**: DDU1231 - **Nhóm lớp**: 01  
**Giảng viên hướng dẫn**: TS. Đỗ Như Tài  

---

## 👥 Thành viên nhóm thực hiện
| STT | Họ và tên | Mã số sinh viên | Vai trò & Phần phụ trách |
| :---: | :--- | :---: | :--- |
| 1 | **Phan Đồng Minh Quân** (Nhóm trưởng) | **3123580038** | Business Understanding, Thiết kế kiến trúc PyTorch MLP, Hyperparameter Tuning, Weighted Ensemble, Tổng hợp báo cáo & kho mã nguồn. |
| 2 | **Võ Minh Tiến** | **3123580053** | Data Understanding, Phân tích EDA, Xử lý Outliers & Missing Values, Domain Feature Engineering, Mô hình so sánh (XGBoost, CatBoost, Lasso) & Trích xuất Submission. |

---

## 📌 Giới thiệu dự án
Dự án được xây dựng dựa trên cuộc thi [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) trên nền tảng Kaggle và nghiên cứu khoa học của **Dean De Cock (2011)** về bộ dữ liệu Ames Housing Dataset.

Mục tiêu chính là xây dựng một **End-to-End Machine Learning Pipeline** chuẩn mực theo quy trình quốc tế **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*), với trọng tâm là mô hình học sâu **Multi-Layer Perceptron (MLP)** lập trình bằng **PyTorch**, đồng thời mở rộng thử nghiệm và kết hợp có trọng số (**Weighted Ensemble**) với các thuật toán Gradient Tree Boosting (XGBoost, CatBoost) và Linear Regularization (Lasso, Ridge).

---

## 🚀 Điểm nổi bật & Cải tiến kỹ thuật
1. **Phát hiện & Xử lý ngoại lệ chuẩn khoa học theo Dean De Cock (2011)**:
   - Thay vì loại bỏ cứng toàn bộ nhà có diện tích lớn, nhóm loại bỏ đúng 2 điểm ngoại lệ dị thường (`GrLivArea > 4000 sqft` và `SalePrice < $300,000` - Id 524 và Id 1299). Đây là những giao dịch thanh lý đặc biệt làm sai lệch mô hình hồi quy.
2. **Kỹ nghệ đặc trưng nghiệp vụ bất động sản (Domain Feature Engineering)**:
   - `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`: Tổng diện tích sàn xây dựng hoàn thiện (đạt tương quan kỷ lục **r = 0.8329** với giá nhà).
   - `TotalBath = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath`: Tổng số phòng tắm có trọng số.
   - `HouseAge = YrSold - YearBuilt`: Tuổi căn nhà tại thời điểm giao dịch.
   - `RemodAge = YrSold - YearRemodAdd`: Tuổi thọ kể từ lần cải tạo, sửa chữa gần nhất.
   - `IsRemodeled`: Cờ nhị phân phân biệt nhà nguyên bản và nhà đã sửa chữa.
   - `TotalPorchSF`: Tổng diện tích hiên và ban công.
3. **Chiến lược điền khuyết theo bản chất ngữ nghĩa dữ liệu**:
   - Các cột `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu`, `Garage...`, `Bsmt...` nếu mang giá trị `NA` được gán chính xác là `'None'` (không có tiện ích).
   - `LotFrontage` được điền bằng trung vị (`median`) của từng khu phố (`Neighborhood`) tương ứng.
4. **Kiến trúc mạng nơ-ron PyTorch MLP hiện đại**:
   - Sử dụng hàm kích hoạt **`LeakyReLU(0.1)`** chống hiện tượng tê liệt nơ-ron ("Dying ReLU").
   - Kỹ thuật điều chuẩn **`Dropout(0.20 - 0.25)`** và chuẩn hóa trọng số **`weight_decay`** (L2 penalty) trong bộ tối ưu **`Adam`**.
5. **Weighted Ensemble tối ưu hóa có ràng buộc**:
   - Tối ưu hóa trọng số kết hợp 4 mô hình độc lập bằng thuật toán SLSQP, với ràng buộc mô hình Deep Learning MLP đóng góp ít nhất 12% trọng số:
   $$\hat{y}_{\text{ensemble}} = 0.1200 \times \hat{y}_{\text{MLP}} + 0.1500 \times \hat{y}_{\text{XGBoost}} + 0.5243 \times \hat{y}_{\text{CatBoost}} + 0.2057 \times \hat{y}_{\text{Lasso}}$$

---

## 📊 Bảng tổng kết kết quả thực nghiệm (Validation Set)

| Mô hình | Họ thuật toán | Validation RMSE (log1p) | Hệ số $R^2$ | Validation MAE (USD) | Xếp hạng |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MLP Baseline** | Neural Network (PyTorch) | 0.15180 | 0.852 | $17,420 | 7 |
| **Ridge Regression** | Linear Model (L2 Penalty) | 0.12839 | 0.894 | $14,980 | 6 |
| **MLP Tuned (Exp 3)** | Neural Network (PyTorch) | **0.14043** | 0.878 | $15,630 | 5 |
| **XGBoost Regressor** | Gradient Tree Boosting | 0.11965 | 0.912 | $13,850 | 4 |
| **Lasso Regression** | Linear Model (L1 Penalty) | 0.11911 | 0.913 | $13,790 | 3 |
| **CatBoost Regressor** | Symmetric Tree Boosting | 0.11448 | 0.919 | $13,090 | 2 |
| 🏆 **Weighted Ensemble (Champion)** | **Ensemble 4 Mô hình (MLP + XGB + CB + Lasso)** | **0.11426** | **0.91971** | **$13,049.89** | **1 (Tốt nhất)** |

---

## 📁 Cấu trúc thư mục dự án
```text
Lab02/
│
├── train.csv                                  # Tập dữ liệu huấn luyện gốc (1460 dòng, 81 cột)
├── test.csv                                   # Tập dữ liệu kiểm tra gốc (1459 dòng, 80 cột)
├── data_description.txt                       # Mô tả chi tiết 79 thuộc tính từ Dean De Cock
├── sample_submission.csv                      # Định dạng mẫu nộp bài Kaggle
│
├── build_lab02.py                             # Kịch bản thực thi toàn bộ pipeline huấn luyện & xuất kết quả
├── generate_notebook.py                       # Kịch bản khởi tạo Jupyter Notebook hoàn chỉnh
├── create_report.py                           # Kịch bản tự động biên soạn Report.docx và Report.pdf
│
├── House_price_pytorch.ipynb                  # Jupyter Notebook hoàn chỉnh có sẵn outputs & đồ thị
├── Report.docx                                # Báo cáo hoàn chỉnh chuẩn mực CRISP-DM (Microsoft Word)
├── Report.pdf                                 # Báo cáo định dạng PDF xuất bản chính thức
├── submission.csv                             # Tệp dự đoán 1459 mẫu kiểm thử nộp lên Kaggle
│
├── experiments/
│   └── house_mlp_v1/
│       ├── best_model.pt                      # Trọng số PyTorch MLP Baseline
│       ├── best_tuned_model.pt                # Trọng số PyTorch MLP Tuned tốt nhất
│       ├── params.json                        # Cấu hình tham số thực nghiệm tập trung
│       ├── tuning_results.csv                 # Nhật ký kết quả 4 thực nghiệm tuning
│       ├── evaluation_metrics.csv             # Bảng chỉ số đánh giá chi tiết
│       └── images/                            # Toàn bộ hình ảnh đồ thị phục vụ báo cáo
│           ├── figure_001.png                 # Biểu đồ giá trị khuyết
│           ├── figure_002.png                 # Phân phối biến mục tiêu SalePrice & log1p
│           ├── figure_003.png                 # Ngoại lệ dị thường theo Dean De Cock (2011)
│           ├── figure_004.png                 # Tương quan GrLivArea sau khi làm sạch
│           ├── figure_005.png                 # Tương quan đặc trưng mới sinh TotalSF (r = 0.833)
│           ├── figure_006.png                 # Phân bổ kích thước Train/Validation/Test
│           ├── figure_007.png                 # Đường cong Loss MLP Baseline qua 120 epochs
│           ├── figure_008.png                 # So sánh Validation RMSE giữa các cấu hình Tuning MLP
│           ├── figure_009.png                 # So sánh hiệu năng tất cả các mô hình
│           ├── figure_010.png                 # Biểu đồ phân tán giá thực tế vs dự đoán & Phân phối phần dư
│           └── Trainning_Time.png             # So sánh thời gian huấn luyện
│
└── TaiLieu/
    ├── decock.pdf                             # Bài báo khoa học của tác giả Dean De Cock (2011)
    └── House Prices_ Advanced Regression Techniques - Rishabh Nimje.pdf
```

---

## ⚙️ Hướng dẫn cài đặt và chạy thử nghiệm

### 1. Cài đặt môi trường Python & các thư viện:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install torch torchvision torchaudio
pip install xgboost catboost lightgbm python-docx docx2pdf
```

### 2. Tái lập toàn bộ Pipeline thí nghiệm:
```bash
cd Lab02
python build_lab02.py
```
Lệnh trên sẽ tự động:
1. Nạp và làm sạch dữ liệu (loại bỏ đúng 2 ngoại lệ).
2. Xử lý giá trị khuyết và sinh 9 đặc trưng mới.
3. Huấn luyện mạng PyTorch MLP Baseline và lưu `best_model.pt`.
4. Thực hiện Tuning 4 cấu hình và lưu `best_tuned_model.pt`.
5. Huấn luyện XGBoost, CatBoost, Lasso, Ridge.
6. Tối ưu hóa trọng số Weighted Ensemble.
7. Xuất toàn bộ đồ thị chất lượng cao vào `experiments/house_mlp_v1/images/`.
8. Tạo file nộp bài `submission.csv` cho 1,459 căn nhà tập test.

### 3. Xem và chạy Jupyter Notebook:
Mở tệp `House_price_pytorch.ipynb` bằng VS Code hoặc Jupyter Lab:
```bash
jupyter notebook House_price_pytorch.ipynb
```

---

## 📚 Tài liệu tham khảo
1. Dean De Cock (2011). *"Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project"*. Journal of Statistics Education, Volume 19, Number 3.
2. Kaggle Competition: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
3. PyTorch Documentation: [Deep Learning with PyTorch](https://pytorch.org/tutorials/).
4. Tianqi Chen and Carlos Guestrin (2016). *"XGBoost: A Scalable Tree Boosting System"*. ACM SIGKDD.
5. Liudmila Prokhorenkova et al. (2018). *"CatBoost: unbiased boosting with categorical features"*. NeurIPS.
