import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "experiments", "house_mlp_v1", "images")
DOCX_PATH = os.path.join(BASE_DIR, "Report.docx")
PDF_PATH = os.path.join(BASE_DIR, "Report.pdf")

doc = docx.Document()

# Set page margins (2.5 cm all around)
for section in doc.sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

# Colors
COLOR_PRIMARY = RGBColor(27, 54, 93)     # Deep Navy
COLOR_SECONDARY = RGBColor(16, 124, 65)  # Forest Green
COLOR_DARK = RGBColor(33, 37, 41)        # Charcoal

def set_cell_background(cell, hex_color):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_table_borders(table):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="CCCCCC"/>'
        f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="CCCCCC"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="E0E0E0"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def add_heading_1(text):
    h = doc.add_paragraph()
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.keep_with_next = True
    run = h.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(15)
    run.font.bold = True
    run.font.color.rgb = COLOR_PRIMARY
    return h

def add_heading_2(text):
    h = doc.add_paragraph()
    h.paragraph_format.space_before = Pt(10)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True
    run = h.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = COLOR_SECONDARY
    return h

def add_heading_3(text):
    h = doc.add_paragraph()
    h.paragraph_format.space_before = Pt(6)
    h.paragraph_format.space_after = Pt(2)
    h.paragraph_format.keep_with_next = True
    run = h.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.bold = True
    run.font.italic = True
    run.font.color.rgb = COLOR_DARK
    return h

def add_p(text, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    p.alignment = align
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = COLOR_DARK
    return p

def add_bullet(text, bold_prefix=""):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(12)
        r_pre.font.bold = True
        r_pre.font.color.rgb = COLOR_PRIMARY
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.color.rgb = COLOR_DARK
    return p

def add_image_with_caption(img_filename, caption_text, width=Inches(5.8)):
    path = os.path.join(IMG_DIR, img_filename)
    if os.path.exists(path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(2)
        p_img.add_run().add_picture(path, width=width)
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(8)
        r_cap = p_cap.add_run(caption_text)
        r_cap.font.name = 'Times New Roman'
        r_cap.font.size = Pt(10.5)
        r_cap.font.italic = True
        r_cap.font.color.rgb = RGBColor(100, 100, 100)

print("Building Report Cover Page...")

# ==================== TRANG BÌA ====================
p_ubnd = doc.add_paragraph()
p_ubnd.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_ubnd.paragraph_format.space_after = Pt(2)
r = p_ubnd.add_run("ỦY BAN NHÂN DÂN THÀNH PHỐ HỒ CHÍ MINH\nTRƯỜNG ĐẠI HỌC SÀI GÒN\nKHOA CÔNG NGHỆ THÔNG TIN")
r.font.name = 'Times New Roman'
r.font.size = Pt(12.5)
r.font.bold = True
r.font.color.rgb = COLOR_PRIMARY

# Dòng phân cách
p_line = doc.add_paragraph()
p_line.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_line.paragraph_format.space_after = Pt(50)
r_line = p_line.add_run("--------------------***--------------------")
r_line.font.bold = True

p_title = doc.add_paragraph()
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_title.paragraph_format.space_after = Pt(8)
r_title = p_title.add_run("BÁO CÁO BÀI TẬP NHÓM LAB02")
r_title.font.name = 'Times New Roman'
r_title.font.size = Pt(19)
r_title.font.bold = True
r_title.font.color.rgb = COLOR_PRIMARY

p_sub = doc.add_paragraph()
p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_sub.paragraph_format.space_after = Pt(70)
r_sub = p_sub.add_run("ĐỀ TÀI: QUI TRÌNH MÁY HỌC CHO DỰ ĐOÁN GIÁ NHÀ\n(HOUSE PRICES - ADVANCED REGRESSION TECHNIQUES)")
r_sub.font.name = 'Times New Roman'
r_sub.font.size = Pt(14)
r_sub.font.bold = True
r_sub.font.color.rgb = COLOR_SECONDARY

# Thông tin môn học & nhóm
p_info = doc.add_paragraph()
p_info.alignment = WD_ALIGN_PARAGRAPH.LEFT
p_info.paragraph_format.left_indent = Inches(1.2)
p_info.paragraph_format.space_after = Pt(90)
p_info.paragraph_format.line_spacing = 1.3

def add_info_line(p, label, val):
    r1 = p.add_run(label)
    r1.font.name = 'Times New Roman'
    r1.font.size = Pt(12.5)
    r1.font.bold = True
    r2 = p.add_run(val + "\n")
    r2.font.name = 'Times New Roman'
    r2.font.size = Pt(12.5)

add_info_line(p_info, "Học phần: ", "Học sâu (Deep Learning)")
add_info_line(p_info, "Mã học phần: ", "858048  -  Nhóm lớp: 01")
add_info_line(p_info, "Lớp: ", "DDU1231")
add_info_line(p_info, "Giảng viên hướng dẫn: ", "TS. Đỗ Như Tài")
add_info_line(p_info, "Sinh viên thực hiện: ", "")
add_info_line(p_info, "   1. Phan Đồng Minh Quân", " - MSSV: 3123580038 (Nhóm trưởng)")
add_info_line(p_info, "   2. Võ Minh Tiến", " - MSSV: 3123580053")

p_bot = doc.add_paragraph()
p_bot.alignment = WD_ALIGN_PARAGRAPH.CENTER
r_bot = p_bot.add_run("Thành phố Hồ Chí Minh, Tháng 09 Năm 2026")
r_bot.font.name = 'Times New Roman'
r_bot.font.size = Pt(12)
r_bot.font.bold = True

doc.add_page_break()

# ==================== TRANG PHÂN CÔNG CÔNG VIỆC ====================
add_heading_1("BẢNG PHÂN CÔNG NHIỆM VỤ THỰC HIỆN")
add_p("Bài tập nhóm Lab02 được các thành viên thực hiện nghiêm túc, phân định rõ ràng các trách nhiệm theo từng giai đoạn trong chu trình chuẩn CRISP-DM:")

table_task = doc.add_table(rows=1, cols=4)
table_task.alignment = WD_TABLE_ALIGNMENT.CENTER
set_table_borders(table_task)
hdr_cells = table_task.rows[0].cells
headers = ["Họ và tên", "MSSV", "Phần phụ trách & Nhiệm vụ chi tiết", "Sản phẩm đầu ra"]
for i, h in enumerate(headers):
    hdr_cells[i].text = h
    set_cell_background(hdr_cells[i], "1B365D")
    p = hdr_cells[i].paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in p.runs:
        r.font.name = 'Times New Roman'
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(11)

tasks_data = [
    (
        "Phan Đồng Minh Quân\n(Nhóm trưởng)",
        "3123580038",
        "- Xác định bài toán kinh doanh (Business Understanding) & nghiên cứu bài báo Dean De Cock (2011).\n"
        "- Thiết kế kiến trúc mạng Multi-Layer Perceptron (MLP) trong PyTorch sử dụng LeakyReLU và Dropout.\n"
        "- Thực hiện khảo sát & Hyperparameter Tuning cho mạng MLP (4 cấu hình thực nghiệm).\n"
        "- Xây dựng bộ tối ưu hóa trọng số Weighted Ensemble.\n"
        "- Thiết kế cấu trúc báo cáo chuẩn CRISP-DM và tổng hợp mã nguồn kho GitHub.",
        "Mô hình PyTorch: best_model.pt, best_tuned_model.pt.\n"
        "Tệp nhật ký tuning: tuning_results.csv.\n"
        "Mã nguồn pipeline: build_lab02.py, House_price_pytorch.ipynb.\n"
        "Báo cáo toàn diện Lab02 (.docx & .pdf)."
    ),
    (
        "Võ Minh Tiến",
        "3123580053",
        "- Khám phá dữ liệu (Data Understanding), phân tích giá trị khuyết và phân phối lệch của biến mục tiêu SalePrice.\n"
        "- Tiền xử lý dữ liệu (Data Preparation), xử lý ngoại lệ GrLivArea theo De Cock và Domain-Specific Imputation.\n"
        "- Xây dựng các đặc trưng mới (Feature Engineering: TotalSF, TotalBath, HouseAge, RemodAge, TotalPorchSF).\n"
        "- Huấn luyện các mô hình bổ sung: XGBoost, CatBoost, Lasso và Ridge Regression.\n"
        "- Đánh giá mô hình toàn diện (Evaluation), phân tích sai số Residuals và trích xuất file nộp Kaggle submission.csv.",
        "Tập dữ liệu tiền xử lý chuẩn hóa.\n"
        "Mô hình so sánh: XGBoost, CatBoost, Lasso, Ridge.\n"
        "Tệp đánh giá: evaluation_metrics.csv.\n"
        "Tệp nộp bài: submission.csv (1,459 mẫu kiểm thử).\n"
        "Đồ thị trực quan hóa phân phối và phần dư."
    )
]

for row in tasks_data:
    r_cells = table_task.add_row().cells
    for i, txt in enumerate(row):
        r_cells[i].text = txt
        p = r_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for r in p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(10.5)

add_p("")
doc.add_page_break()

# ==================== LỜI MỞ ĐẦU & LỜI CẢM ƠN ====================
add_heading_1("LỜI MỞ ĐẦU")
add_p("Trong lĩnh vực Trí tuệ nhân tạo và Khoa học Dữ liệu, bài toán hồi quy định giá bất động sản là một trong những ứng dụng thực tế điển hình và quan trọng nhất. Dữ liệu giá nhà không chỉ chịu ảnh hưởng từ các yếu tố trực quan như diện tích sàn hay số lượng phòng tắm, mà còn phản ánh sự tương tác phức tạp giữa chất lượng vật liệu, vị trí không gian địa lý, lịch sử sửa chữa cải tạo và niên hạn công trình.")
add_p("Cuộc thi 'House Prices - Advanced Regression Techniques' trên nền tảng Kaggle cung cấp bộ dữ liệu Ames Housing Dataset do Giáo sư Dean De Cock biên soạn năm 2011 để thay thế cho bộ dữ liệu Boston Housing kinh điển. Với 79 biến giải thích phản ánh hầu hết các khía cạnh của nhà ở dân cư, bài toán đặt ra thách thức lớn về xử lý dữ liệu chiều cao, giá trị khuyết mang tính ngữ nghĩa, phân phối lệch và các tương tác phi tuyến tính phức tạp.")
add_p("Trong khuôn khổ Bài tập nhóm Lab02 của học phần Học sâu (Deep Learning), nhóm chúng em xây dựng một quy trình máy học hoàn chỉnh từ đầu đến cuối (End-to-End ML Pipeline) tuân thủ quy chuẩn quốc tế CRISP-DM. Trọng tâm của đồ án là thiết kế, tối ưu mạng nơ-ron nhân tạo nhiều tầng Multi-Layer Perceptron (MLP) được lập trình hoàn toàn bằng thư viện PyTorch, đồng thời mở rộng nghiên cứu phối hợp với các giải thuật Gradient Boosting và Linear Regularization để đạt được độ chính xác và khả năng tổng quát hóa tối ưu.")

add_heading_1("LỜI CẢM ƠN")
add_p("Nhóm chúng em xin bày tỏ lòng biết ơn sâu sắc và chân thành nhất đến Thầy TS. Đỗ Như Tài - Giảng viên phụ trách học phần Học sâu (Deep Learning) - Khoa Công nghệ Thông tin, Trường Đại học Sài Gòn.")
add_p("Trong suốt quá trình học tập trên lớp, Thầy đã tận tâm truyền thụ những kiến thức nền tảng vững chắc và chuyên sâu về mạng nơ-ron nhân tạo, cơ chế lan truyền ngược (backpropagation), các giải thuật tối ưu hóa đạo hàm riêng, cũng như phương pháp luận khoa học trong việc phân tích và đánh giá mô hình học máy. Những bài giảng sinh động, định hướng tư duy thực nghiệm và sự chỉ dẫn nhiệt tình của Thầy là nguồn động lực và kim chỉ nam quý báu giúp nhóm hoàn thành đồ án này.")
add_p("Mặc dù nhóm đã nỗ lực vận dụng kiến thức và thực nghiệm khoa học nghiêm túc, báo cáo chắc chắn khó tránh khỏi những hạn chế nhất định. Chúng em rất mong nhận được những nhận xét, đóng góp quý giá từ Thầy để nhóm tiếp tục trau dồi và nâng cao năng lực chuyên môn trong các học phần tiếp theo.")
add_p("Kính chúc Thầy dồi dào sức khỏe, niềm vui và luôn gặt hái nhiều thành công rực rỡ trong sự nghiệp nghiên cứu và giảng dạy!", italic=True)

doc.add_page_break()

# ==================== MỤC LỤC ====================
add_heading_1("MỤC LỤC")
toc_items = [
    ("BẢNG PHÂN CÔNG NHIỆM VỤ THỰC HIỆN", "2"),
    ("LỜI MỞ ĐẦU", "3"),
    ("LỜI CẢM ƠN", "3"),
    ("1. HIỂU BÀI TOÁN (BUSINESS UNDERSTANDING)", "5"),
    ("   1.1. Bối cảnh bài toán và Mục tiêu kinh doanh", "5"),
    ("   1.2. Mục tiêu kỹ thuật và Quy trình chuẩn CRISP-DM", "5"),
    ("   1.3. Tiêu chí đánh giá định lượng (Evaluation Metrics)", "6"),
    ("2. KHÁM PHÁ DỮ LIỆU (DATA UNDERSTANDING)", "7"),
    ("   2.1. Tổng quan tập dữ liệu Ames Housing", "7"),
    ("   2.2. Khảo sát giá trị khuyết (Missing Values Analysis)", "8"),
    ("   2.3. Phân phối biến mục tiêu SalePrice & Phép biến đổi Logarit", "9"),
    ("   2.4. Khảo sát điểm ngoại lệ (Outliers) theo paper Dean De Cock (2011)", "10"),
    ("3. TIỀN XỬ LÝ DỮ LIỆU & KỸ NGHỆ ĐẶC TRƯNG (DATA PREPARATION)", "12"),
    ("   3.1. Loại bỏ các điểm ngoại lệ dị thường", "12"),
    ("   3.2. Chiến lược xử lý giá trị khuyết theo ngữ nghĩa miền (Domain Imputation)", "13"),
    ("   3.3. Kỹ nghệ đặc trưng mới (Domain Feature Engineering)", "14"),
    ("   3.4. Mã hóa One-Hot, Phân chia tập dữ liệu & Chuẩn hóa StandardScaler", "16"),
    ("   3.5. Chuyển đổi cấu trúc PyTorch Tensors", "17"),
    ("4. XÂY DỰNG MÔ HÌNH (MODELING)", "18"),
    ("   4.1. Kiến trúc mạng Multi-Layer Perceptron (MLP) trong PyTorch", "18"),
    ("   4.2. Huấn luyện mô hình MLP Baseline", "19"),
    ("   4.3. Tinh chỉnh siêu tham số (Hyperparameter Tuning) cho mạng MLP", "21"),
    ("   4.4. Xây dựng các mô hình bổ sung (XGBoost, CatBoost, Lasso, Ridge)", "23"),
    ("   4.5. Xây dựng mô hình kết hợp trọng số (Weighted Ensemble Model)", "25"),
    ("5. ĐÁNH GIÁ MÔ HÌNH (EVALUATION)", "27"),
    ("   5.1. So sánh hiệu năng giữa các mô hình", "27"),
    ("   5.2. Đánh giá sai số trên đơn vị tiền tệ thực tế (USD)", "28"),
    ("   5.3. Phân tích phần dư (Residuals Analysis) và Kiểm định giả định", "29"),
    ("6. TRIỂN KHAI VÀ NỘP BÀI (DEPLOYMENT)", "30"),
    ("   6.1. Quy trình suy luận (Inference Pipeline) trên tập Test", "30"),
    ("   6.2. Kiểm tra tính toàn vẹn và xuất tệp submission.csv", "31"),
    ("   6.3. Hướng dẫn cài đặt và tái lập kết quả thực nghiệm", "31"),
    ("KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN", "32"),
    ("TÀI LIỆU THAM KHẢO", "33")
]

for title, pg in toc_items:
    p_toc = doc.add_paragraph()
    p_toc.paragraph_format.space_after = Pt(2)
    p_toc.paragraph_format.line_spacing = 1.15
    r_t = p_toc.add_run(title)
    r_t.font.name = 'Times New Roman'
    r_t.font.size = Pt(11)
    if not title.startswith("   "):
        r_t.font.bold = True
        r_t.font.color.rgb = COLOR_PRIMARY
    dots = " " + "." * max(5, 75 - len(title)) + " "
    r_d = p_toc.add_run(dots)
    r_d.font.name = 'Times New Roman'
    r_d.font.color.rgb = RGBColor(180, 180, 180)
    r_p = p_toc.add_run(pg)
    r_p.font.name = 'Times New Roman'
    r_p.font.bold = True
    r_p.font.size = Pt(11)

doc.add_page_break()

# ==================== NỘI DUNG CHÍNH ====================
print("Writing Section 1: Business Understanding...")
add_heading_1("1. HIỂU BÀI TOÁN (BUSINESS UNDERSTANDING)")

add_heading_2("1.1. Bối cảnh bài toán và Mục tiêu kinh doanh")
add_p("Định giá chính xác bất động sản là mắt xích then chốt trong nền kinh tế thị trường, ảnh hưởng trực tiếp đến người mua nhà, nhà đầu tư, các tổ chức tín dụng cấp hạn mức thế chấp và cơ quan quản lý thuế nhà đất. Theo truyền thống, việc thẩm định giá dựa trên các chuyên viên định giá thông qua so sánh thủ công một vài căn nhà tương đồng trong khu vực. Tuy nhiên, phương pháp thủ công này bộc lộ nhiều điểm yếu: tốn kém thời gian, chi phí cao, mang tính chủ quan cảm tính và khó mở rộng quy mô khi lượng giao dịch bùng nổ.")
add_p("Hệ thống thẩm định giá tự động (Automated Valuation Model - AVM) ứng dụng Học sâu và Học máy cho phép khai thác đồng thời hàng chục đặc tính vi mô của từng ngôi nhà, từ đó đưa ra mức định giá khách quan, tức thời và chuẩn xác. Mục tiêu kinh doanh của dự án là xây dựng một mô hình máy học có khả năng dự đoán chính xác giá bán căn nhà ('SalePrice') tại thành phố Ames, bang Iowa, Hoa Kỳ, giảm thiểu tối đa sai số định giá để phục vụ công tác tư vấn tài chính bất động sản.")

add_heading_2("1.2. Mục tiêu kỹ thuật và Quy trình chuẩn CRISP-DM")
add_p("Về mặt kỹ thuật, đây là bài toán Học có giám sát (Supervised Learning) thuộc lớp bài toán Hồi quy (Regression Problem) với không gian đặc trưng dạng bảng (Tabular Data) nhiều chiều. Nhóm tuân thủ nghiêm ngặt quy trình chuẩn công nghiệp CRISP-DM (Cross-Industry Standard Process for Data Mining) bao gồm 6 giai đoạn liên tục:")
add_bullet(" Nắm bắt yêu cầu định giá bất động sản, cấu trúc dữ liệu và tiêu chuẩn đánh giá.", "1. Business Understanding:")
add_bullet(" Khảo sát phân bố, cấu trúc dữ liệu khuyết, ma trận tương quan và các ngoại lệ dị thường.", "2. Data Understanding:")
add_bullet(" Làm sạch dữ liệu, điền khuyết theo ngữ nghĩa miền, sinh đặc trưng mới, chuẩn hóa và chuyển đổi Tensor.", "3. Data Preparation:")
add_bullet(" Xây dựng mạng nơ-ron MLP PyTorch, huấn luyện baseline, tuning siêu tham số và kết hợp đa mô hình.", "4. Modeling:")
add_bullet(" Đánh giá định lượng trên log1p và đơn vị USD, phân tích biểu đồ phần dư và độ tin cậy.", "5. Evaluation:")
add_bullet(" Đóng gói quy trình suy luận, tạo file submission.csv chuẩn Kaggle và tài liệu hóa kho lưu trữ.", "6. Deployment:")

add_heading_2("1.3. Tiêu chí đánh giá định lượng (Evaluation Metrics)")
add_p("Theo quy định của cuộc thi Kaggle 'House Prices: Advanced Regression Techniques', thước đo chính thức để xếp hạng là Root Mean Squared Error (RMSE) tính trên logarit tự nhiên của giá bán:")
add_p("RMSE_log = sqrt( (1 / N) * sum_{i=1}^N ( log(1 + y_i) - log(1 + y_hat_i) )^2 )", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
add_p("Việc tính sai số trên thang logarit mang ý nghĩa kinh tế cực kỳ quan trọng: Sai số tương đối (phần trăm) giữa căn nhà đắt tiền và căn nhà giá rẻ được cân bằng công bằng. Ví dụ, việc dự đoán lệch $20,000 đối với căn nhà trị giá $100,000 (sai lệch 20%) sẽ bị phạt nặng hơn nhiều so với việc dự đoán lệch $20,000 đối với căn biệt thự trị giá $1,000,000 (chỉ sai lệch 2%).")
add_p("Bên cạnh đó, để phục vụ góc nhìn kinh doanh thực tế, nhóm bổ sung hai độ đo theo giá trị tiền tệ thực tế (USD):")
add_bullet(" Độ lệch tuyệt đối trung bình tính bằng tiền thực tế.", "Mean Absolute Error (USD): ")
add_bullet(" Hệ số xác định phản ánh tỷ lệ phương sai của biến mục tiêu được giải thích bởi mô hình.", "Hệ số R^2 (R-squared): ")

# ==================== SECTION 2 ====================
print("Writing Section 2: Data Understanding...")
add_heading_1("2. KHÁM PHÁ DỮ LIỆU (DATA UNDERSTANDING)")

add_heading_2("2.1. Tổng quan tập dữ liệu Ames Housing")
add_p("Bộ dữ liệu Ames Housing do Dean De Cock thu thập từ văn phòng Thẩm định viên thành phố Ames, Iowa trong giai đoạn từ năm 2006 đến 2010. Dữ liệu bao gồm 2 tập chính:")
add_bullet(" gồm 1,460 dòng quan sát và 81 cột (gồm 79 biến thuộc tính mô tả căn nhà, cột định danh Id và biến mục tiêu SalePrice).", "Tập huấn luyện (train.csv):")
add_bullet(" gồm 1,459 dòng và 80 cột (chứa đầy đủ các thuộc tính nhưng đã bị ẩn cột SalePrice để phục vụ việc chấm điểm khách quan).", "Tập kiểm thử (test.csv):")
add_p("Các thuộc tính được phân thành 4 nhóm chính: Nhóm định lượng liên tục (diện tích các tầng, diện tích đất, mặt tiền), nhóm định lượng rời rạc (năm xây dựng, số phòng ngủ, số phòng tắm), nhóm định danh phân loại (loại móng, khu phố, kiểu mái) và nhóm thứ bậc định tính (chất lượng tổng thể, điều kiện hoàn thiện từ Poor đến Excellent).")

add_heading_2("2.2. Khảo sát giá trị khuyết (Missing Values Analysis)")
add_p("Kiểm tra sơ bộ trên tập train phát hiện có 19 thuộc tính có giá trị khuyết (NaN/Null). Hình 1 thể hiện 20 thuộc tính có số lượng giá trị khuyết nhiều nhất trong tập dữ liệu.")

add_image_with_caption("figure_001.png", "Hình 1: Top 20 thuộc tính có số lượng giá trị bị khuyết nhiều nhất trong tập Train")

add_p("Phân tích kỹ lưỡng tài liệu mô tả data_description.txt chỉ ra rằng: Đại đa số các giá trị 'NA' trong tập dữ liệu này không phải là dữ liệu bị mất ngẫu nhiên (Missing Completely at Random - MCAR), mà mang ý nghĩa cấu trúc (Structural Missing Data). Ví dụ:")
add_bullet(" 99.52% giá trị mang giá trị 'NA' vì hầu hết các ngôi nhà bình dân tại Ames không có hồ bơi.", "PoolQC (Chất lượng hồ bơi):")
add_bullet(" 96.30% mang giá trị 'NA' vì ngôi nhà không có tiện ích bổ sung như nhà kho phụ hay thang máy.", "MiscFeature:")
add_bullet(" 93.77% không có lối đi phụ phía sau nhà.", "Alley (Lối đi hẻm):")
add_bullet(" 47.26% mang giá trị 'NA' vì căn nhà không có lò sưởi.", "FireplaceQu (Chất lượng lò sưởi):")
add_bullet(" 5.55% mang giá trị 'NA' vì căn nhà không có garage xe hơi.", "GarageType, GarageFinish, GarageQual, GarageCond:")
add_p("Do đó, việc xử lý vội vàng bằng cách loại bỏ các cột này hoặc điền mode/mean đơn thuần là một sai lầm nghiêm trọng làm biến dạng bản chất vật lý của căn nhà.")

add_heading_2("2.3. Phân phối biến mục tiêu SalePrice & Phép biến đổi Logarit")
add_p("Khảo sát phân phối của biến mục tiêu SalePrice cho thấy giá trị nằm trong khoảng từ $34,900 đến $755,000 với giá trị trung bình là $180,921 và trung vị là $163,000. Phân phối thể hiện độ lệch dương (Skewness = 1.88) và độ nhọn cao (Kurtosis = 6.54) với chiếc đuôi dài kéo dài về phía các căn biệt thự hạng sang.")

add_image_with_caption("figure_002.png", "Hình 2: Phân phối của biến SalePrice trước (trái) và sau khi áp dụng phép biến đổi log1p (phải)")

add_p("Mạng nơ-ron sâu và các thuật toán học máy dựa trên khoảng cách rất nhạy cảm với dữ liệu lệch lớn, do hàm mất mát Mean Squared Error (MSE) sẽ bị chi phối quá mức bởi các căn nhà đắt tiền. Bằng cách áp dụng phép biến đổi log1p:")
add_p("y = log(1 + SalePrice)", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
add_p("Hệ số bất đối xứng Skewness giảm mạnh từ 1.88 xuống chỉ còn 0.12 (gần như đạt chuẩn hoàn hảo). Đồ thị bên phải Hình 2 chứng minh phân phối sau biến đổi tuân theo quy luật phân phối chuẩn Gauss, tạo điều kiện thuận lợi tối đa cho quá trình lan truyền ngược gradient.")

add_heading_2("2.4. Khảo sát điểm ngoại lệ (Outliers) theo paper Dean De Cock (2011)")
add_p("Biến GrLivArea (Ground Living Area - diện tích sinh hoạt trên mặt đất) là một trong những đặc trưng có mối tương quan mạnh nhất với giá bán. Tuy nhiên, khi vẽ biểu đồ phân tán GrLivArea vs SalePrice, ta phát hiện những điểm dữ liệu dị thường nghiêm trọng.")

add_image_with_caption("figure_003.png", "Hình 3: Phát hiện 2 điểm ngoại lệ dị thường theo khuyến nghị khoa học của Dean De Cock (2011)")

add_p("Trong bài báo khoa học xuất bản trên Journal of Statistics Education (2011), Giáo sư Dean De Cock đã phân tích rõ 2 căn nhà đặc biệt (Id 524 với diện tích 4,676 sqft bán giá $184,750 và Id 1299 với diện tích 5,642 sqft bán giá $160,000):")
add_p("Đây là những giao dịch đất đai bán một phần hoặc bất động sản nông nghiệp với các điều kiện thanh lý đặc biệt ('abnormal partial sales'), hoàn toàn không đại diện cho thị trường nhà ở tiêu chuẩn. Hai điểm dữ liệu này nếu giữ lại sẽ tạo ra đòn bẩy hồi quy (high leverage points) cực lớn, làm lệch hoàn toàn đường mặt phẳng dự đoán của mạng nơ-ron. Ngược lại, hai căn nhà có diện tích trên 4,000 sqft khác nhưng có giá bán cao tương ứng trên $700,000 (Id 692 và 1183) là những căn biệt thự siêu sang hợp lệ và cần được giữ lại.")

# ==================== SECTION 3 ====================
print("Writing Section 3: Data Preparation...")
add_heading_1("3. TIỀN XỬ LÝ DỮ LIỆU & KỸ NGHỆ ĐẶC TRƯNG (DATA PREPARATION)")

add_heading_2("3.1. Loại bỏ các điểm ngoại lệ dị thường")
add_p("Thực hiện đúng khuyến nghị của Dean De Cock (2011), nhóm loại bỏ chính xác 2 điểm quan sát ngoại lệ dị thường thỏa mãn điều kiện:")
add_p("(GrLivArea > 4000) AND (SalePrice < 300000)", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
add_p("Tập dữ liệu huấn luyện sau khi lọc còn lại đúng 1,458 mẫu quan sát sạch và sẵn sàng cho các công đoạn tiếp theo. Đồ thị phân tán sau khi làm sạch thể hiện mối quan hệ tuyến tính đồng biến rõ ràng, đẹp mắt giữa diện tích sinh hoạt và giá nhà (Hình 4).")

add_image_with_caption("figure_004.png", "Hình 4: Tương quan GrLivArea và SalePrice sau khi loại bỏ 2 điểm ngoại lệ dị thường")

add_heading_2("3.2. Chiến lược xử lý giá trị khuyết theo ngữ nghĩa miền (Domain Imputation)")
add_p("Nhóm xây dựng hàm tiền xử lý preprocess_housing thực thi trên tập dữ liệu gộp chung cả Train và Test để đảm bảo không gian biến và cách xử lý đồng nhất tuyệt đối:")
add_bullet(" Đối với các biến PoolQC, MiscFeature, Alley, Fence, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, MasVnrType, toàn bộ giá trị NaN được thay thế bằng chuỗi 'None' thể hiện sự vắng mặt của tiện ích.", "1. Nhóm thuộc tính có NA mang nghĩa 'None':")
add_bullet(" Đối với các biến GarageCars, GarageArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath, BsmtHalfBath, MasVnrArea, GarageYrBlt, giá trị NaN được thay bằng số 0.", "2. Nhóm thuộc tính số lượng hầm và garage:")
add_bullet(" LotFrontage (chiều rộng mặt tiền tiếp giáp đường) phụ thuộc theo quy hoạch phân lô từng khu phố. Nhóm gom nhóm theo Neighborhood và điền khuyết bằng trung vị (median) của chính khu phố đó.", "3. Nhóm thuộc tính mặt tiền LotFrontage:")
add_bullet(" Các biến phân loại còn lại được điền bằng giá trị xuất hiện nhiều nhất (mode), các biến số còn lại được điền bằng trung vị (median) của toàn tập.", "4. Các biến khác:")

add_heading_2("3.3. Kỹ nghệ đặc trưng mới (Domain Feature Engineering)")
add_p("Kỹ nghệ đặc trưng là bước đột phá mang lại sự khác biệt lớn về hiệu năng cho bài làm của nhóm. Thay vì chỉ phụ thuộc vào các thuộc tính thô ban đầu, nhóm thiết kế thêm 9 đặc trưng nghiệp vụ bất động sản chuyên sâu:")
add_bullet(" Tổng diện tích sàn xây dựng có hoàn thiện = TotalBsmtSF + 1stFlrSF + 2ndFlrSF. Đây là chỉ số diện tích thực tế toàn diện nhất của căn nhà.", "TotalSF:")
add_bullet(" Tổng số phòng tắm có trọng số = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath.", "TotalBath:")
add_bullet(" Tuổi của ngôi nhà tại thời điểm giao dịch = YrSold - YearBuilt.", "HouseAge:")
add_bullet(" Số năm kể từ lần cải tạo, sửa chữa cuối cùng = YrSold - YearRemodAdd.", "RemodAge:")
add_bullet(" Biến cờ nhị phân = 1 nếu YearRemodAdd != YearBuilt (nhà đã qua sửa chữa), ngược lại = 0.", "IsRemodeled:")
add_bullet(" Tổng diện tích hiên và ban công = WoodDeckSF + OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch.", "TotalPorchSF:")
add_bullet(" Các cờ nhị phân chỉ thị sự hiện diện của Garage, Tầng hầm và Lò sưởi.", "HasGarage, HasBsmt, HasFireplace:")

add_p("Đặc biệt, đặc trưng mới sinh TotalSF đạt hệ số tương quan tuyến tính Pearson với SalePrice lên tới r = 0.8329 (vượt trội hơn bất kỳ đặc trưng đơn lẻ nào trong bộ dữ liệu gốc, bao gồm cả OverallQual r = 0.7958 và GrLivArea r = 0.7350). Hình 5 minh họa trực quan sự gắn kết chặt chẽ này.")

add_image_with_caption("figure_005.png", "Hình 5: Mối quan hệ tương quan mạnh mẽ giữa đặc trưng TotalSF mới sinh và giá bán SalePrice (r = 0.833)")

add_heading_2("3.4. Mã hóa One-Hot, Phân chia tập dữ liệu & Chuẩn hóa StandardScaler")
add_p("Sau khi tạo các đặc trưng, toàn bộ biến phân loại được chuyển đổi thành các cột nhị phân thông qua hàm pd.get_dummies(..., drop_first=True). Sau khi mã hóa, không gian đặc trưng mở rộng thành 267 chiều.")
add_p("Tập huấn luyện (1,458 mẫu) được phân chia thành 2 tập độc lập theo tỷ lệ 75% - 25% với random_state = 42 cố định:")
add_bullet(" 1,093 mẫu dùng để cập nhật trọng số mạng nơ-ron và các mô hình.", "Tập Train (75%):")
add_bullet(" 365 mẫu dùng để đánh giá khách quan sai số, giám sát overfitting và tìm trọng số ensemble.", "Tập Validation (25%):")
add_bullet(" 1,459 mẫu kiểm thử nộp bài Kaggle.", "Tập Kaggle Test:")

add_image_with_caption("figure_006.png", "Hình 6: Phân bổ số lượng mẫu giữa các tập dữ liệu Train, Validation và Kaggle Test")

add_p("Để mạng nơ-ron hội tụ nhanh và ổn định, toàn bộ ma trận đặc trưng được chuẩn hóa theo phân phối chuẩn tắc (Z-score Scaling) sử dụng StandardScaler. Bộ chuẩn hóa được fit duy nhất trên tập Train, sau đó dùng để transform cho Validation và Kaggle Test nhằm tuân thủ nguyên tắc ngăn chặn rò rỉ thông tin (Data Leakage).")

add_heading_2("3.5. Chuyển đổi cấu trúc PyTorch Tensors")
add_p("Dữ liệu sau khi chuẩn hóa được ép kiểu sang số thực 32-bit (torch.float32) và đóng gói vào cấu trúc TensorDataset cùng DataLoader với batch_size = 32 và shuffle = True trên tập huấn luyện, sẵn sàng cho việc tính toán tensor trên đồ thị tính toán đạo hàm tự động của PyTorch.")

# ==================== SECTION 4 ====================
print("Writing Section 4: Modeling...")
add_heading_1("4. XÂY DỰNG MÔ HÌNH (MODELING)")

add_heading_2("4.1. Kiến trúc mạng Multi-Layer Perceptron (MLP) trong PyTorch")
add_p("Theo yêu cầu cốt lõi của đề tài, nhóm xây dựng mô hình mạng nơ-ron nhân tạo nhiều tầng MLP bằng thư viện PyTorch (lớp kế thừa nn.Module). Kiến trúc mạng được thiết kế đặc thù cho dữ liệu dạng bảng nhiều chiều:")
add_bullet(" 267 nơ-ron (tương ứng 267 đặc trưng đã được chuẩn hóa).", "Tầng vào (Input Layer):")
add_bullet(" Tầng ẩn 1 (48 nơ-ron) -> Kích hoạt LeakyReLU(0.1) -> Dropout(0.20) -> Tầng ẩn 2 (24 nơ-ron) -> Kích hoạt LeakyReLU(0.1).", "Các tầng ẩn (Hidden Layers):")
add_bullet(" 1 nơ-ron đầu ra dự đoán giá trị log1p(SalePrice).", "Tầng ra (Output Layer):")

add_p("So với cấu trúc mạng dùng ReLU thông thường, việc sử dụng LeakyReLU(negative_slope=0.1) giải quyết triệt để vấn đề 'Dying ReLU' (hiện tượng các nơ-ron bị tê liệt do gradient bằng 0 khi đầu vào âm), đảm bảo dòng gradient luôn lưu thông thông suốt qua toàn bộ mạng.")

add_heading_2("4.2. Huấn luyện mô hình MLP Baseline")
add_p("Mô hình MLP Baseline được huấn luyện qua 120 Epochs với cấu hình:")
add_bullet(" Adam (Adaptive Moment Estimation) với learning_rate = 0.003.", "Bộ tối ưu hóa:")
add_bullet(" MSELoss() tính trên sai lệch giữa nhãn thực tế và đầu ra mạng.", "Hàm mất mát:")
add_bullet(" weight_decay = 0.003 (tương đương chuẩn L2 Regularization để phạt các trọng số quá lớn).", "Trọng số phạt L2:")

add_image_with_caption("figure_007.png", "Hình 7: Đồ thị hàm mất mát Train Loss và Validation Loss của mô hình MLP Baseline qua 120 Epochs")

add_p("Hình 7 ghi nhận quá trình hội tụ mượt mà của mô hình. Cả Train Loss và Validation Loss giảm nhanh chóng trong 30 epoch đầu tiên và duy trì ổn định, không xuất hiện hiện tượng phân kỳ (divergence). Trọng số mô hình có Validation Loss nhỏ nhất được tự động lưu lại vào tệp best_model.pt. Kết quả Validation RMSE của mô hình baseline đạt 0.15180 trong thời gian huấn luyện chỉ 4.5 giây.")

add_heading_2("4.3. Tinh chỉnh siêu tham số (Hyperparameter Tuning) cho mạng MLP")
add_p("Để tối ưu hóa năng lực dự đoán của mạng MLP, nhóm thiết kế kịch bản thử nghiệm lưới (Grid Search Experiment) khảo sát 4 cấu hình siêu tham số độc lập về số nơ-ron tầng ẩn, tỷ lệ Dropout, tốc độ học và trọng số suy giảm L2:")

table_tune = doc.add_table(rows=1, cols=7)
table_tune.alignment = WD_TABLE_ALIGNMENT.CENTER
set_table_borders(table_tune)
t_hdr = table_tune.rows[0].cells
t_titles = ["Thí nghiệm", "Hidden 1", "Hidden 2", "Dropout", "Learning Rate", "Weight Decay", "Validation RMSE"]
for i, h in enumerate(t_titles):
    t_hdr[i].text = h
    set_cell_background(t_hdr[i], "1B365D")
    p = t_hdr[i].paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in p.runs:
        r.font.name = 'Times New Roman'
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(10)

tune_data = [
    ("Exp 1", "48", "24", "0.15", "0.0030", "0.003", "0.14839"),
    ("Exp 2", "48", "24", "0.20", "0.0035", "0.004", "0.14981"),
    ("Exp 3 (Tối ưu)", "64", "32", "0.25", "0.0030", "0.004", "0.14043"),
    ("Exp 4", "32", "16", "0.20", "0.0035", "0.004", "0.14468")
]

for row in tune_data:
    r_cells = table_tune.add_row().cells
    for i, txt in enumerate(row):
        r_cells[i].text = txt
        p = r_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if "Tối ưu" in row[0]:
            set_cell_background(r_cells[i], "E8F5E9")
        for r in p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(10)
            if "Tối ưu" in row[0]:
                r.font.bold = True

add_image_with_caption("figure_008.png", "Hình 8: Biểu đồ so sánh Validation RMSE giữa 4 cấu hình Hyperparameter Tuning MLP")

add_p("Kết quả thực nghiệm cho thấy cấu hình Exp 3 với kiến trúc [64, 32] kết hợp tỷ lệ Dropout 0.25 và Weight Decay 0.004 đạt hiệu năng tốt nhất với Validation RMSE hạ xuống mức 0.14043 (cải thiện đáng kể so với mức 0.15180 của Baseline). Trọng số tối ưu này được lưu vào tệp best_tuned_model.pt làm thành phần đại diện cho họ mô hình Deep Learning.")

add_heading_2("4.4. Xây dựng các mô hình bổ sung (XGBoost, CatBoost, Lasso, Ridge)")
add_p("Để tạo ra sự đa dạng về cơ chế học thuật toán, nhóm triển khai thêm 4 mô hình bổ sung thuộc 2 họ thuật toán khác:")
add_bullet(" Cây quyết định tăng cường độ dốc với n_estimators = 800, max_depth = 3, learning_rate = 0.03, subsample = 0.8, colsample_bytree = 0.8. Đạt Validation RMSE = 0.11965.", "1. XGBoost Regressor:")
add_bullet(" Thuật toán tăng cường xử lý đặc trưng đối xứng tối tân với iterations = 900, learning_rate = 0.03, depth = 5. Đạt Validation RMSE = 0.11448.", "2. CatBoost Regressor:")
add_bullet(" Mô hình hồi quy tuyến tính với chuẩn phạt L1 (LassoCV) tự động tinh chỉnh alpha tối ưu = 0.00307 qua 5-Fold CV. Đạt Validation RMSE = 0.11911.", "3. Lasso Regression (L1):")
add_bullet(" Mô hình tuyến tính với chuẩn phạt L2 (RidgeCV). Đạt Validation RMSE = 0.12839.", "4. Ridge Regression (L2):")

add_heading_2("4.5. Xây dựng mô hình kết hợp trọng số (Weighted Ensemble Model)")
add_p("Ensemble học máy dựa trên nguyên lý đa dạng hóa sai số: Các mô hình có cơ chế hoạt động khác nhau (mạng nơ-ron MLP, cây quyết định boosting và mô hình tuyến tính) sẽ mắc sai số ở các quan sát khác nhau. Khi kết hợp trung bình có trọng số, các sai số độc lập này sẽ triệt tiêu lẫn nhau, giúp phương sai của dự đoán tổng thể giảm mạnh:")
add_p("y_hat_ensemble = sum_{k=1}^4 w_k * y_hat_k,   sao cho sum w_k = 1, w_k >= 0", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
add_p("Nhóm sử dụng giải thuật tối ưu hóa có ràng buộc SLSQP (Sequential Least Squares Programming) thông qua thư viện scipy.optimize.minimize để tìm bộ trọng số w tối thiểu hóa hàm RMSE trên tập Validation, với ràng buộc mô hình Deep Learning MLP đóng góp ít nhất 12%:")
add_bullet(" Đóng góp 12.00% trọng số.", "Mô hình PyTorch MLP (Deep Learning):")
add_bullet(" Đóng góp 15.00% trọng số.", "Mô hình XGBoost:")
add_bullet(" Đóng góp 52.43% trọng số.", "Mô hình CatBoost:")
add_bullet(" Đóng góp 20.57% trọng số.", "Mô hình Lasso Regression (Linear L1):")

add_p("Mô hình Weighted Ensemble tổng hợp đạt được Validation RMSE xuất sắc ở mức 0.11426, đánh bại hoàn toàn mọi mô hình đơn lẻ.")

# ==================== SECTION 5 ====================
print("Writing Section 5: Evaluation...")
add_heading_1("5. ĐÁNH GIÁ MÔ HÌNH (EVALUATION)")

add_heading_2("5.1. So sánh hiệu năng giữa các mô hình")
add_p("Bảng tổng kết dưới đây trình bày toàn bộ kết quả đánh giá thực nghiệm trên cùng một tập dữ liệu Validation độc lập (365 mẫu quan sát):")

table_eval = doc.add_table(rows=1, cols=4)
table_eval.alignment = WD_TABLE_ALIGNMENT.CENTER
set_table_borders(table_eval)
e_hdr = table_eval.rows[0].cells
e_titles = ["Mô hình", "Họ phương pháp", "Validation RMSE (log1p)", "Xếp hạng"]
for i, h in enumerate(e_titles):
    e_hdr[i].text = h
    set_cell_background(e_hdr[i], "1B365D")
    p = e_hdr[i].paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in p.runs:
        r.font.name = 'Times New Roman'
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(10)

eval_rows = [
    ("MLP Baseline", "Neural Network (PyTorch)", "0.15180", "7"),
    ("Ridge Regression", "Linear Model (L2 Penalty)", "0.12839", "6"),
    ("MLP Tuned (Exp 3)", "Neural Network (PyTorch)", "0.14043", "5"),
    ("XGBoost Regressor", "Gradient Tree Boosting", "0.11965", "4"),
    ("Lasso Regression", "Linear Model (L1 Penalty)", "0.11911", "3"),
    ("CatBoost Regressor", "Symmetric Tree Boosting", "0.11448", "2"),
    ("Weighted Ensemble (Champion)", "Ensemble 4 Mô hình (MLP + XGB + CB + Lasso)", "0.11426", "1 (Tốt nhất)")
]

for row in eval_rows:
    r_cells = table_eval.add_row().cells
    for i, txt in enumerate(row):
        r_cells[i].text = txt
        p = r_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if "Champion" in row[0]:
            set_cell_background(r_cells[i], "E8F5E9")
        for r in p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(10)
            if "Champion" in row[0]:
                r.font.bold = True

add_image_with_caption("figure_009.png", "Hình 9: So sánh trực quan Validation RMSE giữa các mô hình học máy và mô hình kết hợp Weighted Ensemble")

add_heading_2("5.2. Đánh giá sai số trên đơn vị tiền tệ thực tế (USD)")
add_p("Để chuyển đổi kết quả học máy thành thông tin hữu ích trong thực tế kinh doanh bất động sản, nhóm thực hiện nghịch đảo hàm logarit bằng công thức y_pred_usd = exp(y_hat) - 1. Kết quả thu được trên 365 căn nhà thuộc tập Validation:")
add_bullet(" Mô hình giải thích được 91.97% sự biến thiên của giá nhà thực tế trên thị trường.", "Hệ số xác định R^2: 0.91971.")
add_bullet(" Trung bình mô hình chỉ lệch khoảng $13,049 USD trên mỗi căn nhà (tương đương sai số chỉ khoảng 7.2% so với giá trị trung bình $180,921 của thị trường).", "Sai số tuyệt đối trung bình: MAE = $13,049.89 USD.")
add_bullet(" Sai số toàn phương đạt $19,194.85 USD, kiểm soát rất tốt các ngoại lệ giá lớn.", "Căn bậc hai sai số toàn phương: RMSE = $19,194.85 USD.")

add_heading_2("5.3. Phân tích phần dư (Residuals Analysis) và Kiểm định giả định")
add_p("Hình 10 phân tích chi tiết chất lượng dự đoán của mô hình:")

add_image_with_caption("figure_010.png", "Hình 10: Biểu đồ phân tán giá thực tế vs giá dự đoán (trái) và Phân phối sai số phần dư Residuals (phải)")

add_bullet(" Các điểm dữ liệu nằm bám rất sát đường chéo lý tưởng màu đỏ (y = x) từ phân khúc nhà bình dân ($50,000) đến các căn biệt thự sang trọng ($400,000+), chứng tỏ mô hình có tính tuyến tính và độ tin cậy đồng đều.", "Biểu đồ phân tán (trái):")
add_bullet(" Phân phối sai số phần dư đối xứng hoàn hảo quanh trục 0, có dạng hình chuông chuẩn tắc và không bị lệch một phía. Điều này khẳng định mô hình không bị thiên vị hệ thống (No systematic bias), thỏa mãn đầy đủ các giả định cơ bản của lý thuyết hồi quy thống kê.", "Biểu đồ phần dư (phải):")

add_image_with_caption("Trainning_Time.png", "Hình 11: So sánh thời gian huấn luyện giữa các mô hình (MLP, XGBoost, CatBoost, Lasso)")

# ==================== SECTION 6 ====================
print("Writing Section 6: Deployment...")
add_heading_1("6. TRIỂN KHAI VÀ NỘP BÀI (DEPLOYMENT)")

add_heading_2("6.1. Quy trình suy luận (Inference Pipeline) trên tập Test")
add_p("Quy trình suy luận tự động được thực hiện qua các bước khép kín:")
add_bullet(" Nạp tập dữ liệu kiểm tra 1,459 mẫu từ test.csv.", "Bước 1 - Nạp dữ liệu:")
add_bullet(" Áp dụng hàm preprocess_housing để điền khuyết theo quy tắc miền và sinh 9 đặc trưng mới.", "Bước 2 - Tiền xử lý:")
add_bullet(" Áp dụng bộ scaler đã fit từ tập huấn luyện để chuẩn hóa đặc trưng.", "Bước 3 - Chuẩn hóa:")
add_bullet(" Lần lượt đưa ma trận đặc trưng qua 4 mô hình thành phần: PyTorch MLP (best_tuned_model.pt), XGBoost, CatBoost và Lasso.", "Bước 4 - Dự đoán đơn lẻ:")
add_bullet(" Tính tổng có trọng số: y_pred_log = 0.12*MLP + 0.15*XGB + 0.5243*CatBoost + 0.2057*Lasso.", "Bước 5 - Ensemble kết hợp:")
add_bullet(" Chuyển đổi ngược về giá tiền đô-la Mỹ: SalePrice = np.expm1(y_pred_log) và làm tròn 2 chữ số thập phân.", "Bước 6 - Nghịch đảo logarit:")

add_heading_2("6.2. Kiểm tra tính toàn vẹn và xuất tệp submission.csv")
add_p("Tệp kết quả cuối cùng submission.csv được khởi tạo thành công với cấu trúc hoàn chỉnh theo yêu cầu Kaggle:")
add_bullet(" Đúng 1,459 dòng dữ liệu và 1 dòng tiêu đề (tổng cộng 1,460 dòng).", "Kích thước:")
add_bullet(" Gồm đúng 2 cột bắt buộc là Id và SalePrice.", "Cấu trúc cột:")
add_bullet(" Hoàn toàn không có giá trị Null, NaN hay vô cùng (Infinite).", "Kiểm tra tính toàn vẹn:")

table_sub = doc.add_table(rows=1, cols=2)
table_sub.alignment = WD_TABLE_ALIGNMENT.CENTER
set_table_borders(table_sub)
s_hdr = table_sub.rows[0].cells
s_hdr[0].text = "Id"
s_hdr[1].text = "SalePrice (Dự đoán USD)"
for c in s_hdr:
    set_cell_background(c, "1B365D")
    p = c.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in p.runs:
        r.font.name = 'Times New Roman'
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

sample_sub_data = [
    ("1461", "$121,233.97"),
    ("1462", "$160,740.27"),
    ("1463", "$184,370.85"),
    ("1464", "$196,631.37"),
    ("1465", "$189,646.22"),
    ("1466", "$176,092.02"),
    ("1467", "$182,008.63"),
    ("1468", "$166,072.24"),
    ("1469", "$189,648.90"),
    ("1470", "$121,672.50")
]

for row in sample_sub_data:
    r_cells = table_sub.add_row().cells
    r_cells[0].text = row[0]
    r_cells[1].text = row[1]
    for c in r_cells:
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(10)

add_p("")
add_heading_2("6.3. Hướng dẫn cài đặt và tái lập kết quả thực nghiệm")
add_p("Toàn bộ mã nguồn dự án được lưu trữ tại kho GitHub chính thức của sinh viên:")
add_p("https://github.com/phandongminhquan-sgu/Hoc_sau_BT_Phan-Dong-Minh-Quan.git", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
add_p("Để tái lập toàn bộ kết quả, người dùng chỉ cần clone repository và thực thi lệnh duy nhất:")
add_p("python build_lab02.py", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

# ==================== KẾT LUẬN ====================
print("Writing Conclusion & References...")
add_heading_1("KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN")
add_heading_2("Kết luận đạt được")
add_p("Sau quá trình nghiên cứu, triển khai và đánh giá nghiêm túc, nhóm đã hoàn thành toàn diện tất cả các mục tiêu đề ra cho Bài tập nhóm Lab02:")
add_bullet(" Xây dựng hoàn chỉnh quy trình máy học chuẩn 6 bước CRISP-DM từ khâu thấu hiểu bài toán đến tạo sản phẩm nộp bài Kaggle.", "1. Quy trình chuẩn mực:")
add_bullet(" Đề xuất thành công 9 đặc trưng bất động sản mới, trong đó TotalSF có tương quan kỷ lục r = 0.833 với giá nhà.", "2. Kỹ nghệ đặc trưng đột phá:")
add_bullet(" Xây dựng mô hình PyTorch MLP tùy biến với LeakyReLU và Dropout, huấn luyện và tuning thành công đạt Validation RMSE = 0.14043.", "3. Học sâu PyTorch:")
add_bullet(" Kết hợp tối ưu giữa Deep Learning MLP, Gradient Boosting (XGBoost, CatBoost) và Linear Regularization (Lasso) đưa Validation RMSE về mức 0.11426, R^2 đạt 91.97% và MAE ở mức $13,049 USD.", "4. Weighted Ensemble xuất sắc:")

add_heading_2("Hướng phát triển tương lai")
add_bullet(" Thử nghiệm các kiến trúc chuyên sâu cho dữ liệu bảng hiện đại như TabNet, FT-Transformer (Feature Tokenizer Transformer) hoặc SAINT.", "1. Mở rộng kiến trúc:")
add_bullet(" Ứng dụng kỹ thuật xếp tầng Stacking Ensemble sử dụng mô hình meta-learner phi tuyến để học cách kết hợp các dự đoán linh hoạt hơn theo từng phân khúc giá.", "2. Stacking nâng cao:")
add_bullet(" Tích hợp tọa độ địa lý GPS thực tế của từng khu phố tại thành phố Ames để đưa vào các mạng tích chập không gian (Spatial GCN) nhằm mô hình hóa hiệu ứng tương tác láng giềng.", "3. Tọa độ không gian:")

# ==================== TÀI LIỆU THAM KHẢO ====================
add_heading_1("TÀI LIỆU THAM KHẢO")
refs = [
    "Dean De Cock (2011). 'Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project'. Journal of Statistics Education, Volume 19, Number 3.",
    "Kaggle Competition. 'House Prices: Advanced Regression Techniques'. URL: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques",
    "PyTorch Documentation. 'Deep Learning with PyTorch: A 60 Minute Blitz'. URL: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html",
    "Tianqi Chen and Carlos Guestrin (2016). 'XGBoost: A Scalable Tree Boosting System'. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.",
    "Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, Andrey Gulin (2018). 'CatBoost: unbiased boosting with categorical features'. Advances in Neural Information Processing Systems (NeurIPS).",
    "Hastie, T., Tibshirani, R., & Friedman, J. (2009). 'The Elements of Statistical Learning: Data Mining, Inference, and Prediction'. Springer Science & Business Media."
]
for i, ref in enumerate(refs):
    add_bullet(ref, f"[{i+1}] ")

# Save Document
doc.save(DOCX_PATH)
print(f"Report.docx created successfully: {DOCX_PATH}")

# Convert to PDF
try:
    from docx2pdf import convert
    print("Converting Report.docx to Report.pdf via docx2pdf...")
    convert(DOCX_PATH, PDF_PATH)
    print(f"Report.pdf created successfully: {PDF_PATH}")
except Exception as e:
    print(f"docx2pdf note: {e}")
