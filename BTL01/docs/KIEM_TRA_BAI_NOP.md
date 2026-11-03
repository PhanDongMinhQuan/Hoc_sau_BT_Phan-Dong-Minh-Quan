# Đối chiếu yêu cầu BTL01

Nhóm 1 — Phan Đồng Minh Quân (3123580038), Võ Minh Tiến (3123580053).

## Yêu cầu và minh chứng

| Yêu cầu | Minh chứng trong thư mục BTL01 |
|---|---|
| Đúng Gaussian, mu, C, miền x | src/data.py; báo cáo mục 1–2; tests/test_core.py |
| Train/validation/test độc lập; test đúng 100 mẫu | data/train.csv, validation.csv, test.csv; manifest.json |
| Scatter 3D, bề mặt thật, phân bố | figures/01_data_3d.png, 02_distributions.png; results/data_statistics.csv |
| MLP tùy biến bằng PyTorch | src/model.py, GaussianMLP kế thừa nn.Module |
| DataLoader, MSELoss, Adam, vòng backward | src/training.py; báo cáo mục 4–6 |
| Loss từng epoch, checkpoint theo validation | results/runs/*/history.csv, best.pt, metrics.json |
| Thực nghiệm A: 200 đến 2000, bước 200 | results/experiment_A.csv; figures/03_experiment_A.png |
| Thực nghiệm B: 1–10 và mở rộng 16,32,64 | results/experiment_B.csv; figures/04_experiment_B.png |
| Thực nghiệm C: 1–4 tầng, loss và thời gian | results/experiment_C.csv; figures/05_experiment_C.png, 06_depth_convergence.png |
| Ba seed, trung bình và độ lệch chuẩn | results/all_runs.csv, summary.csv; seeds 38,53,91 |
| Chọn cấu hình trước khi đánh giá test | results/selection.json; src/experiments.py |
| MSE, MAE, RMSE, R² và hình dự đoán | results/final_evaluation.json; test_predictions.csv; figures/08–10 |
| Báo cáo Word và PDF | report/BTL01_Nhom1_BaoCao.docx và .pdf, 26 trang; 20 trang nội dung chính |
| Slide PPTX và PDF | slides/BTL01_Nhom1_ThuyetTrinh.pptx và .pdf, 15 slide |
| Hướng dẫn và môi trường | README.md, requirements.txt, requirements-lock.txt, results/environment.json |
| Thành viên, nhiệm vụ, tổng tỷ lệ 100% | Bìa, bảng phân công báo cáo, README; mỗi thành viên 50% |
| Chuẩn bị demo và giải thích code | docs/HUONG_DAN_BAO_VE.md; demo.py; verify.py |

## Kiểm tra kỹ thuật

- Tám kiểm thử tự động: công thức Gaussian độc lập, dữ liệu tái lập/không giao nhau, chỉ số trên ví dụ biết trước, khôi phục checkpoint, từ chối test khi chưa chốt lựa chọn, gradient so với sai phân hữu hạn, ngân sách đối chứng và từ chối checkpoint đại diện bị thay đổi.
- Đối chiếu lại 75 checkpoint với số liệu đã lưu; kiểm tra đủ 30.000 dòng epoch, 81 mục cấu hình-seed và 27 dòng tổng hợp. Các cấu hình trùng giữa A/B/C dùng chung run, không tăng số seed thống kê.
- Đối chiếu thêm 15 checkpoint mới và 21 mục đối chứng train/validation; kiểm tra đúng 12.800 bước và loss cuối. Tổng 90 run riêng, 45.600 dòng lịch sử epoch.
- Kiểm tra số mẫu các vùng cộng đủ 100, MSE có trọng số theo vùng khớp MSE chung, bootstrap đủ 9.999 lần/chỉ số và đúng checkpoint gốc.
- Huấn luyện lại checkpoint đại diện [16,16,16], seed 91, đủ 400 epoch trên cùng máy/thư viện: trọng số khớp từng phần tử và history.csv khớp từng byte; best epoch 371. Không khẳng định khớp từng bit trên mọi thiết bị hoặc phiên bản khác.
- Báo cáo đã xuất bằng Word, slide đã xuất bằng PowerPoint; kiểm tra bố cục từng trang. Các biểu đồ được dựng từ CSV/checkpoint, không điền lại số liệu bằng tay.

Lệnh kiểm chứng sau khi giải nén, thực hiện trong BTL01:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python verify.py
python demo.py --epochs 20
```

Nếu cần đánh giá lại và tạo lại hình, chạy `python main.py evaluate` và `python main.py plots`. Các lệnh này cập nhật tệp kết quả trong bản đang chạy; nên giữ một bản ZIP gốc để đối chiếu.

## Nhóm rà soát trước khi nộp

- Đối chiếu lần cuối tên, MSSV, lớp DDU1231, giảng viên TS. Đỗ Như Tài và phân công 50%–50%.
- Mỗi bạn tự chạy demo và giải thích được phần phụ trách; tập phân biệt số liệu checkpoint đại diện với trung bình ba seed.
- Tập trình bày trong 10–15 phút và đọc ghi chú từng slide.
- Nhóm trưởng kiểm tra hạn nộp và nơi nộp; hoàn tất phần trình bày theo mức hiểu của nhóm.
- Tên gói: BTL01_Nhom1_PhanDongMinhQuan_3123580038.zip.

Kho mã nguồn được lưu tại thư mục BTL01 trong repo phandongminhquan-sgu/Hoc_sau_BT_Phan-Dong-Minh-Quan. Nhóm trưởng sử dụng ZIP để nộp trên Classroom; việc bàn giao tệp không thay thế phần kiểm tra code và thuyết trình trực tiếp của hai thành viên.
