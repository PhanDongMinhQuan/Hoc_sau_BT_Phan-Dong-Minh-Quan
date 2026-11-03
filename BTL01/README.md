# BTL01 Nhóm 1

Xấp xỉ hàm mật độ Gaussian hai chiều bằng mạng MLP trong PyTorch.

| Thành viên | MSSV | Vai trò | Phân công | Tỷ lệ |
|---|---|---|---|---|
| Phan Đồng Minh Quân | 3123580038 | Nhóm trưởng | Tích hợp mã nguồn, kiểm tra tái lập, báo cáo và trình bày | 50% |
| Võ Minh Tiến | 3123580053 | Thành viên | Dữ liệu, thực nghiệm A/B/C và phân tích kết quả | 50% |

Trường Đại học Sài Gòn, Khoa Toán Ứng Dụng, lớp DDU1231. Giảng viên: TS. Đỗ Như Tài.

Kho mã nguồn: [BTL01 trên GitHub](https://github.com/phandongminhquan-sgu/Hoc_sau_BT_Phan-Dong-Minh-Quan/tree/main/BTL01).

## Bài toán và dữ liệu

Đầu vào x=(x1,x2) lấy mẫu đều trên [-2,2] × [-4,4]. Nhãn là mật độ Gaussian với mu=[0,0], C=[[0.3,-0.5],[-0.5,2]], không thêm nhiễu. Chương trình tự tính công thức giải tích.

Ba tập có 2000 mẫu train, 500 validation và 100 test. Dùng PCG64 với SeedSequence([3123580038,3123580053]) và spawn_key riêng 0,1,2. Dữ liệu riêng được sinh từ đầu. Các cấu hình A dùng những tập con lồng nhau gồm A dòng đầu của tập train cố định.

Chuẩn hóa đầu vào bằng x1/2 và x2/4, dựa vào miền biết trước trong đề. Không ước lượng tham số chuẩn hóa từ validation/test. Nhãn y và mọi chỉ số sai số đều giữ thang đo gốc.

## Môi trường và chạy lại

Python 3.10 trở lên, CPU là lựa chọn mặc định. Phiên bản môi trường thực tế nằm trong results/environment.json và requirements-lock.txt.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python verify.py
```

Chạy lại trọn bộ vào một thư mục mới, giữ nguyên kết quả đã nộp:

```bash
python main.py all --output reproduced --device cpu
```

Chạy từng bước hoặc từng nhóm:

```bash
python main.py prepare --output reproduced
python main.py train --experiment A --output reproduced
python main.py train --experiment B --output reproduced
python main.py train --experiment C --output reproduced
python main.py select --output reproduced
python main.py evaluate --output reproduced
python main.py plots --output reproduced
```

Lệnh train tiếp tục các run còn thiếu nếu bị ngắt; chỉ tái sử dụng một run khi toàn bộ cấu hình giống nhau. Sau select, lựa chọn được đóng băng và train sẽ từ chối ghi thêm vào cùng nghiên cứu. Muốn thay đổi thiết kế, tạo thư mục output mới và thực hiện quy trình riêng.

Đánh giá checkpoint đã nộp và dựng lại hình không cần huấn luyện:

```bash
python main.py evaluate
python main.py plots
```

Chạy demo nhanh, không sửa kết quả chính:

```bash
python demo.py --epochs 20
```

Nếu dùng môi trường Anaconda có nhiều thư viện OpenMP, các điểm vào chương trình đặt MKL_THREADING_LAYER=SEQUENTIAL trước khi import NumPy/PyTorch. Không cần GPU, không cần tài khoản hay tải bộ dữ liệu ngoài. Thời gian thực tế từng run đã được lưu; tốc độ chạy trên máy khác có thể khác.

## Thiết kế thực nghiệm

Thiết lập chung: Tanh, đầu ra Linear, MSELoss, Adam, learning rate 0.002, batch size 64, 400 epoch cố định, 1 CPU thread, seed huấn luyện 38/53/91. Không early stopping, không scheduler, không weight decay. Chọn checkpoint có validation MSE nhỏ nhất trong 400 epoch.

| Nhóm | Biến thay đổi | Các giá trị | Các yếu tố cố định |
|---|---|---|---|
| A | Số mẫu train | 200,400,...,2000 | Một tầng ẩn 16 nơ-ron |
| B | Độ rộng | 1,2,...,10,16,32,64 | Một tầng ẩn, 2000 mẫu |
| C | Độ sâu | 1,2,3,4 | 16 nơ-ron mỗi tầng, 2000 mẫu |

Có 81 mục cấu hình-seed trong ba nhóm, tương ứng 75 lượt huấn luyện độc lập. A=2000, B=16 và C=1 có cùng kiến trúc, dữ liệu và seed nên dùng chung kết quả. Đây không phải thêm lượt lặp thống kê: mỗi cấu hình vẫn có đúng ba seed.

Các tập validation và test được giữ cố định. Chỉ validation được dùng để chọn kiến trúc: so sánh trung bình validation MSE của ba seed; nếu bằng nhau ưu tiên ít tham số rồi ít mẫu train. Trong cấu hình được chọn, checkpoint đại diện là seed có validation MSE thấp nhất. Quy tắc và mã băm kế hoạch được ghi vào results/selection.json trước khi đọc test.

Sau khi đóng băng lựa chọn, tất cả checkpoint được đánh giá trên cùng 100 mẫu test để lập các bảng A/B/C. Không dùng kết quả test để sửa hyperparameter hoặc chọn lại mô hình. Lưới 121×121 chỉ là chẩn đoán bổ sung sau lựa chọn, không thay thế tập test.

Lưu ý: cố định số epoch khiến số bước cập nhật tăng khi A tăng. Kết quả A phản ánh đồng thời lượng dữ liệu và ngân sách cập nhật dưới quy tắc epoch cố định. Ba seed phản ánh biến động khởi tạo/xáo trộn; chưa phản ánh biến động do lấy một bộ dữ liệu mới.

## Cấu trúc thư mục

```text
BTL01/
  main.py, demo.py          Lệnh điều phối và demo ngắn
  verify.py                Đối chiếu checkpoint, log và số liệu đã nộp
  src/                     Dữ liệu, mô hình, huấn luyện, thí nghiệm, đồ thị
  tests/                   Kiểm tra công thức, dữ liệu, checkpoint, quy trình
  data/                    Train/validation/test CSV và mã băm
  results/                 Kế hoạch, lựa chọn, số liệu tổng hợp, kết quả test
    runs/<run_id>/         metrics.json, history.csv, best.pt của từng run
  checkpoints/             selected_model.pt
  figures/                 Hình được tạo từ dữ liệu và kết quả thực tế
  report/                  Báo cáo DOCX và PDF
  slides/                  Thuyết trình PPTX và PDF
  docs/                    Hướng dẫn đọc code, bảo vệ và kiểm tra bài nộp
  assets/                  Logo trường và ghi nguồn
```

## Đọc kết quả

- all_runs.csv: từng cấu hình-seed, cả train/validation/test, thời gian và best epoch.
- summary.csv và experiment_A/B/C.csv: trung bình và độ lệch chuẩn mẫu (ddof=1) qua ba seed.
- runs/*/history.csv: train và validation loss tại cùng trọng số cuối mỗi epoch.
- selection.json: lựa chọn được chốt bằng validation.
- final_evaluation.json: checkpoint đại diện, hai mô hình hằng làm mốc so sánh và chẩn đoán lưới.
- test_predictions.csv: từng đầu vào, y thật, y dự đoán, sai số.

RMSE trung bình được tính bằng trung bình RMSE từng seed. Nó không bắt buộc bằng căn bậc hai của MSE trung bình. Kết quả một checkpoint đại diện cũng khác với trung bình ba seed. Chỉ so sánh các đại lượng cùng định nghĩa.

Lệnh `python verify.py` kiểm tra 75 checkpoint, 30.000 dòng lịch sử epoch, 81 mục cấu hình-seed, 27 dòng tổng hợp và lựa chọn bằng validation. Chỉ số train/validation được tính với nhãn float32 đã chuyển thành tensor trong huấn luyện; nhãn test giữ float64 từ hàm giải tích. Phần kiểm chứng dùng đúng quy ước này, đặc biệt khi đối chiếu R² rất gần 0.

Để chuẩn bị thuyết trình, đọc [hướng dẫn bảo vệ](docs/HUONG_DAN_BAO_VE.md). [Bảng kiểm bài nộp](docs/KIEM_TRA_BAI_NOP.md) liên kết từng yêu cầu với tệp minh chứng.

## Phân tích bổ sung và bản báo cáo hoàn thiện

Báo cáo có 26 trang: 20 trang nội dung chính (trang 4–23), ba trang bìa/tóm tắt/mục lục, một trang tài liệu tham khảo và hai trang phụ lục. Slide có 15 trang.

Hai đối chứng được thiết kế sau thực nghiệm chính, chỉ dùng train/validation và không thay đổi selection.json hoặc checkpoint đã chọn. Chúng không được trình bày như một đánh giá test mới.

| Đối chứng | Cấu hình | Mục đích |
|---|---|---|
| Cùng 12.800 bước optimizer | A=200,1000,2000; hidden=[16]; epoch=3200,800,400 | Phân biệt lượng dữ liệu với ngân sách cập nhật |
| Gần 600 tham số | [152], [22,22], [16,16,16], [13,13,13,13] | Giảm ảnh hưởng của tổng tham số khi so sánh độ sâu |

Có 21 mục cấu hình-seed bổ sung: sáu dùng lại run gốc và 15 lượt huấn luyện mới. Tổng cộng lưu 90 run riêng và 45.600 dòng lịch sử epoch. Không cộng các mục dùng lại thành seed mới. Đối chứng A còn báo cáo loss ở đúng bước cuối để kiểm tra ảnh hưởng của số cơ hội chọn checkpoint.

```bash
# Đối chiếu toàn bộ kết quả đã nộp, không huấn luyện
python verify.py
# Tạo lại đối chứng trong thư mục mới
python supplement.py --output supplemental_rerun
# Tính lại chẩn đoán của checkpoint cố định
python diagnostics.py
```

- supplemental/plan.json ghi thiết kế trước các lượt chạy bổ sung và mã băm lựa chọn gốc.
- supplemental/all_runs.csv và summary.csv lưu train/validation, epoch, số bước, nguồn run. Run mới có history.csv, metrics.json và best.pt trong supplemental/runs.
- diagnostics/width_gaps.csv phân tích khoảng cách train–validation.
- diagnostics/regional_test_errors.csv chia 100 mẫu test theo q=xᵀC⁻¹x.
- diagnostics/diagnostics.json lưu khoảng BCa 95% từ 9.999 lần bootstrap, cùng hai CSV chứa đầy đủ mẫu thống kê. Khoảng này có điều kiện trên checkpoint, không bao gồm bất định huấn luyện và không dùng để chọn mô hình.
- verify.py đối chiếu cả checkpoint bổ sung, số liệu tổng hợp, lỗi theo vùng và nguồn checkpoint của phân tích.

Các giới hạn: một bộ dữ liệu, ba seed, test 100 điểm; đối chứng A không đồng nhất batch cuối/thời gian; đối chứng độ sâu chỉ gần bằng tổng tham số. Cần nghiên cứu mới với tập đánh giá độc lập để chọn lại mô hình theo các phát hiện bổ sung.

## Nguồn tham khảo

1. Đỗ Như Tài. MLP Exercise - Yêu cầu, tài liệu môn Học sâu do giảng viên cung cấp.
2. Goodfellow, Bengio, Courville (2016). Deep Learning, chương 6. https://www.deeplearningbook.org/contents/mlp.html
3. Kingma, Ba (2015). Adam: A Method for Stochastic Optimization. https://arxiv.org/abs/1412.6980
4. PyTorch. Optimizing Model Parameters. https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
5. Bài BTL01 của nhóm trong repo SGHPK0905/DeepLearning_3SV_DDU1231, commit 6f195cdf235cf66a18eb4e2b4ae1f1636a220746: tham khảo cách tổ chức đầu ra. Mã triển khai, bộ dữ liệu và số liệu của bài này được xây dựng/chạy riêng, không sao chép kết quả của repo tham khảo.

6. SciPy. scipy.stats.bootstrap, phương pháp BCa. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
7. PyTorch. Reproducibility. https://docs.pytorch.org/docs/2.15/notes/randomness.html

Ngày kiểm tra các nguồn trực tuyến: 03/11/2026. Phần giới thiệu phương pháp trong báo cáo là diễn giải ngắn; kết quả và nhận xét thực nghiệm dựa vào log đi kèm.
