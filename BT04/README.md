# BT04 - MLP và mạng học sâu

**Sinh viên:** Phan Đồng Minh Quân  
**MSSV:** 3123580038  
**Khoa:** Toán Ứng Dụng - Trường Đại học Sài Gòn  
**Học phần:** Học sâu - 858048 - DDU1231  
**Giảng viên:** TS. Đỗ Như Tài

## Bài nộp

[Tải bài làm Word](./BT04_MLP_PhanDongMinhQuan_3123580038.docx)

Tài liệu gồm 10 trang, giải đủ ba bài tập theo từng bước. Trang bìa có logo SGU, họ tên và MSSV. Công thức trong Word có thể chỉnh sửa.

| Tệp | Nội dung |
|---|---|
| [BT04_MLP_PhanDongMinhQuan_3123580038.docx](./BT04_MLP_PhanDongMinhQuan_3123580038.docx) | Bài làm theo định dạng nộp của giảng viên |
| [DL04_Mang MLP.pdf](./DL04_Mang%20MLP.pdf) | Đề bài do giảng viên cung cấp |
| [kiem_tra_tinh_toan.py](./kiem_tra_tinh_toan.py) | Chương trình đối chiếu các phép tính bằng NumPy |
| [ket_qua_tinh_toan.json](./ket_qua_tinh_toan.json) | Kết quả lan truyền xuôi, gradient và cập nhật trọng số |
| [requirements.txt](./requirements.txt) | Phiên bản thư viện dùng để kiểm tra |

## Nội dung thực hiện

1. **Bài 1:** Tính sigmoid và Binary Cross-Entropy, suy ra gradient theo quy tắc dây chuyền, cập nhật ba trọng số và tính lại dự đoán.
2. **Bài 2:** Biểu diễn đầu vào, trọng số và bias bằng vector/ma trận; tính vector hóa và so sánh với cách tính từng nơ-ron.
3. **Bài 3:** So sánh tanh/sigmoid với ReLU/sigmoid, tính gradient và toàn bộ trọng số sau một bước học.

Ở Bài 3, xem nhãn [0, 1] là one-hot của hai lớp loại trừ nhau. Sau khi tính các hàm kích hoạt theo đề, áp dụng **softmax lên vector sau kích hoạt** rồi tính Cross-Entropy đa lớp. Lựa chọn này xử lý vấn đề tanh/ReLU không bảo đảm đầu ra là xác suất, theo lưu ý ở trang 3 của đề. Chương trình kiểm tra sử dụng đúng định nghĩa này.

## Chạy kiểm tra

Đã chạy với Python 3.12 và NumPy 2.3.5. Từ thư mục gốc của repository:

```bash
python -m pip install -r BT04/requirements.txt
python BT04/kiem_tra_tinh_toan.py
```

Chương trình tính lại kết quả và ghi vào BT04/ket_qua_tinh_toan.json. Các kiểm tra gồm:

- Gradient giải tích so với sai phân trung tâm, bước 1e-6, sai số tuyệt đối nhỏ hơn 1e-9.
- Kết quả phép nhân ma trận khớp cách cộng từng tích.
- Mất mát giảm sau một bước cập nhật trong Bài 1 và cả hai trường hợp của Bài 3.
- Xác suất softmax dương và có tổng bằng 1.

| Trường hợp | Mất mát trước cập nhật | Mất mát sau cập nhật |
|---|---:|---:|
| Bài 1 - sigmoid và BCE | 0.543792604 | 0.525554010 |
| Bài 3 - tanh và sigmoid | 0.651387031 | 0.639136289 |
| Bài 3 - ReLU và sigmoid | 0.689648093 | 0.662638172 |

Các giá trị trong bảng được làm tròn đến 9 chữ số thập phân; chương trình giữ độ chính xác float64 khi tính toán.

## Nguồn

- Đề bài và sơ đồ mạng: tài liệu DL04_Mang MLP.pdf do giảng viên cung cấp.
- Bố cục trình bày tham khảo từ [BT04 của Trương Tuyết Nghi](https://github.com/Monika51204/Deep-Learning/tree/1d2a08f8da475f4ff402f98502ffed391e5ffb04/Tu%E1%BA%A7n%2008). Các phép tính trong bài này được kiểm tra riêng theo định nghĩa hàm mất mát đã nêu.
