# Hướng dẫn đọc code và bảo vệ BTL01

## Lộ trình đọc

1. Đọc README để nắm đầu vào, đầu ra và cách chạy.
2. Mở src/data.py: xác định mu, C, miền x và công thức gaussian. Giải thích vì sao lấy x đều rồi tính y.
3. Mở src/model.py: chỉ ra nn.Module, Linear, Tanh và tầng đầu ra. Tự đếm số tham số của mạng [16].
4. Mở src/training.py: giải thích DataLoader, zero_grad, backward, step, eval, no_grad và lưu trạng thái tốt nhất.
5. Mở src/experiments.py: đọc matrix, freeze_selection và evaluate_final theo thứ tự. Chỉ ra tập test được đọc ở đâu.
6. Mở một results/runs/<run_id>/history.csv, so sánh best_epoch với checkpoint và metrics.json.
7. Đọc bảng A/B/C trong báo cáo, tập giải thích một điểm không theo xu hướng mong đợi.

## Demo khoảng hai phút

Mở terminal ở BTL01, kích hoạt môi trường đã cài thư viện.

```bash
python -m unittest discover -s tests -v
python main.py evaluate
python demo.py --epochs 20
```

Lệnh evaluate nạp mô hình đã huấn luyện; kết quả của checkpoint chính nằm trong final_evaluation.json. Demo 20 epoch chỉ minh họa quá trình huấn luyện trên 200 mẫu và ghi ra quick_demo, nên không kỳ vọng sai số bằng run chính 400 epoch. Không huấn luyện lại cả 75 run trong phần demo ngắn.

Nếu giảng viên muốn xem một thí nghiệm đầy đủ, dùng một thư mục mới:

```bash
python main.py prepare --output reproduced
python main.py train --experiment C --output reproduced
```

## Phân công trình bày dự kiến

- Quân: mở đầu; vòng huấn luyện; lựa chọn mô hình; kết quả cuối; giới hạn và demo.
- Tiến: bài toán; dữ liệu; kiến trúc; thiết kế và kết quả A/B/C.
- Tổng thời gian mục tiêu 10–15 phút. Ghi chú của từng slide có ý chính để luyện nói; không đọc nguyên văn cả slide.

## Câu hỏi thường gặp

**1. Bài này là phân loại hay hồi quy?**

Hồi quy: hai tọa độ vào, một giá trị mật độ liên tục ra. R² không phải accuracy.

**2. Dữ liệu đầu vào có phân phối Gaussian không?**

Không. x được lấy đều trong hình chữ nhật quy định. Gaussian là hàm tính nhãn y tại x. Lấy x trực tiếp từ Gaussian sẽ thay đổi thiết kế dữ liệu.

**3. Tại sao bài nhóm mình có dữ liệu khác nhóm tham khảo?**

Hàm và miền giữ đúng đề. Entropy lấy từ hai MSSV và luồng sinh riêng tạo tọa độ mẫu mới. Mỗi nhóm có thể có các điểm ngẫu nhiên khác nhau dù cùng bài toán.

**4. Vì sao có validation khi đã có test?**

Validation được dùng nhiều lần để chọn epoch và kiến trúc. Test chỉ cung cấp đánh giá sau khi chốt lựa chọn. Nếu liên tục chọn theo test, test không còn là đánh giá độc lập của quy trình lựa chọn.

**5. Tại sao chuẩn hóa bằng x1/2 và x2/4?**

Miền đầu vào đã được cho sẵn. Cách này đưa hai biến về [-1,1] mà không ước lượng thống kê từ validation/test. Nhãn vẫn giữ nguyên.

**6. zero_grad, backward và step khác nhau thế nào?**

zero_grad xóa gradient cũ; backward tính gradient của loss theo trọng số bằng autograd; step dùng gradient và trạng thái Adam để cập nhật trọng số.

**7. model.eval có đồng nghĩa tắt gradient không?**

Không. eval đổi chế độ của những lớp như Dropout/BatchNorm. no_grad mới tắt xây dựng đồ thị gradient. Bài này vẫn dùng cả hai đúng ngữ cảnh dù MLP chỉ có Linear/Tanh.

**8. Tại sao phải clone state_dict tốt nhất?**

Nếu chỉ giữ tham chiếu tới tensor đang được huấn luyện, trọng số có thể tiếp tục thay đổi. detach().cpu().clone() lưu một bản độc lập của epoch được chọn.

**9. Vì sao 81 mục báo cáo mà chỉ 75 run?**

A=2000 với hidden=[16], B=16 và C=1 là một cấu hình. Với cùng seed, dữ liệu và hyperparameter, chúng dùng chung kết quả. Mỗi cấu hình vẫn có đúng ba seed, không đếm các bản dùng lại là lượt lặp mới.

**10. Tăng dữ liệu luôn làm mô hình tốt hơn không?**

Không bảo đảm cho mọi lần chạy. Bài này báo cáo xu hướng thực tế; còn khởi tạo, số bước cập nhật và năng lực mô hình cùng ảnh hưởng. Thí nghiệm A cố định epoch nên số bước cũng tăng theo A.

**11. B=64 vì sao không tốt hơn B=32 trong bài này?**

Kết quả quan sát được trong ngân sách 400 epoch và ba seed cho B=32 có validation MSE trung bình thấp hơn. Chưa đủ bằng chứng để khẳng định nguyên nhân là quá khớp. Có thể cần thêm seed hoặc ngân sách tối ưu, nhưng không điều chỉnh bài hiện tại theo test để kiếm số đẹp hơn.

**12. Mạng sâu hơn sai số tăng có phải tiêu biến gradient không?**

Chưa thể kết luận. Bài chưa đo gradient theo tầng hay làm đối chứng. C tăng cũng làm số tham số thay đổi. Hãy trình bày đây là giả thuyết cần kiểm chứng, không phải nguyên nhân đã xác nhận.

**13. Tại sao dùng đầu ra Linear, có thể dự đoán âm không?**

Linear là lựa chọn thuận tiện cho hồi quy và có thể cho giá trị âm. Bài kiểm tra tỷ lệ âm trên lưới. Nếu ứng dụng cần ràng buộc mật độ dương, có thể khảo sát Softplus riêng.

**14. Vì sao checkpoint MSE khác MSE trung bình ba seed?**

Một số là của một bộ trọng số đại diện; số kia là trung bình của ba mô hình. Checkpoint được chọn bằng validation, không bằng test. Không trộn hai số khi thuyết trình.

**15. RMSE trung bình có bằng sqrt(MSE trung bình) không?**

Không nhất thiết. Mỗi seed tính căn trước, rồi mới lấy trung bình. Phép căn là phi tuyến nên hai thứ tự không tương đương.

**16. Lưới 121×121 có thay test 100 mẫu không?**

Không. Test 100 mẫu là đánh giá theo đề. Lưới chỉ dùng bổ sung để quan sát sai số theo vị trí sau khi lựa chọn đã đóng băng. Max error trên lưới không là chặn chắc chắn cho toàn miền liên tục.

**17. Điểm khác biệt thực chất của bài nhóm mình là gì?**

Bộ mẫu và seed riêng; normalization theo miền; thiết kế baseline16, batch64, lr0.002, 400epoch; lựa chọn dựa trên mean validation; lưu loss mọi epoch; mã băm dữ liệu và kế hoạch; đối chiếu mô hình hằng và phân bố sai số. Các khác biệt phục vụ kiểm chứng, không thay đổi bài toán giảng viên giao.

## Câu hỏi cho phần phân tích bổ sung

**18. Thí nghiệm A đã cố định 400 epoch, tại sao vẫn cần đối chứng?**

Epoch là một lượt qua toàn bộ train. Với batch64, A=200 có 4 bước/epoch, A=2000 có 32 bước/epoch. Vì vậy 400 epoch tương ứng 1.600 và 12.800 cập nhật. Thêm dữ liệu đồng thời thêm cơ hội tối ưu; đối chứng giữ 12.800 bước để kiểm tra phần ảnh hưởng đó.

**19. Cùng số bước có phải hoàn toàn công bằng chưa?**

Chưa. Batch cuối có số mẫu khác nhau, thời gian và số lần xét validation cũng khác. Bài báo cáo thêm loss đúng tại bước cuối để tránh chọn epoch có lợi hơn trong riêng cột đó. Phải nói rõ phần nào được kiểm soát và phần nào chưa.

**20. Vì sao không đổi mô hình chính khi đối chứng bốn tầng có trung bình validation thấp hơn một chút?**

Đối chứng được thiết kế sau khi đã xem đánh giá test chính. Đây là phân tích mô tả nhằm kiểm tra kết luận cũ, không phải một vòng lựa chọn mới. Chênh lệch ba/bốn tầng cũng nhỏ hơn biến động giữa seed. Nếu muốn chọn lại, cần quy trình mới và test độc lập.

**21. Bootstrap ở đây làm gì?**

Giữ nguyên mô hình, lấy mẫu lại có hoàn lại từ 100 sai số test, mỗi lần vẫn lấy 100 phần tử; tính MSE/MAE và lặp 9.999 lần. SciPy dùng BCa để hiệu chỉnh độ chệch và bất đối xứng của phân phối bootstrap. Khoảng MSE khoảng [3,07;5,03]×10⁻⁷ mô tả bất định do 100 điểm test hữu hạn, có điều kiện trên mô hình này. Không phải 95% dự đoán nằm trong khoảng đó.

**22. Ba seed và bootstrap có thay thế nhau không?**

Không. Ba seed thay đổi khởi tạo/xáo trộn huấn luyện trên cùng bộ dữ liệu. Bootstrap thay đổi mẫu đánh giá bằng cách lấy mẫu lại sai số của một mô hình cố định. Hai cách khảo sát hai nguồn biến động khác nhau.

**23. 16,7% điểm dự đoán âm có phải mô hình rất tệ không?**

Cần đọc cả tỷ lệ và độ lớn. Min trên lưới khoảng −0,001573, bằng 0,585% đỉnh thật 0,269. Sai số nhỏ vẫn vi phạm điều kiện mật độ không âm. Bài là xấp xỉ hàm bằng hồi quy; nếu dùng như một mật độ xác suất thực sự thì phải xét ràng buộc đầu ra và chuẩn hóa.

**24. Khi nào có thể nói B=64 quá khớp?**

Cần bằng chứng như train tốt hơn trong khi validation xấu hơn, cùng lịch sử loss và độ ổn định giữa các lần chạy. Trong bài, B=64 tăng cả train lẫn validation so với B=32; vì vậy chưa đủ bằng chứng cho cách giải thích quá khớp. Chỉ hai trong ba seed có validation MSE B=64 cao hơn B=32.

## Bài tập tự kiểm tra trước buổi bảo vệ

Quân tự tìm trong src/training.py nơi gradient được xóa, tính và dùng để cập nhật. Tiến tự tính 609 tham số của mạng ba tầng 16 nơ-ron: (2×16+16) + 2×(16×16+16) + (16×1+1). Đổi vai và giải thích lại bằng lời.

Sau đó, mỗi người mở một history.csv, tìm epoch có validation loss nhỏ nhất, đối chiếu metrics.json và checkpoint. Cuối cùng tập giải thích Hình 11–12 của báo cáo trong một phút mà không đọc nguyên văn.
