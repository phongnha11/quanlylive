import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="PBC-PT Digital Hub - Báo Cáo Lãnh Đạo",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS TÙY CHỈNH ---
st.markdown("""
<style>
    .main { background-color: #f4f6f9; }
    .stMetric {
        background-color: white; padding: 15px; border-radius: 10px;
        border-left: 6px solid #b71c1c; /* Màu đỏ đô thương hiệu */
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    h1, h2, h3 { color: #b71c1c; font-family: 'Segoe UI', sans-serif; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: white; border-radius: 5px; padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffebee; color: #b71c1c; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. HÀM XỬ LÝ DỮ LIỆU ---

# Hàm tạo dữ liệu giả lập (Fallback khi chưa có file thật trên GitHub)
def generate_mock_data(semester_name="Học kỳ I"):
    classes = ['10A1', '10A2', '10A3', '11B1', '11B2', '12C1', '12C2']
    data = []
    np.random.seed(42 if semester_name == "Học kỳ I" else 24) # Seed khác nhau để dữ liệu khác nhau
    
    for i in range(300):
        student_class = np.random.choice(classes)
        # Logic giả lập: A giỏi Tự nhiên, C giỏi Xã hội
        if 'A' in student_class:
            math = np.random.normal(8.0, 1.5)
            lit = np.random.normal(6.5, 1.5)
            eng = np.random.normal(7.0, 2.0)
        elif 'B' in student_class:
            math = np.random.normal(7.0, 1.5)
            lit = np.random.normal(7.0, 1.5)
            eng = np.random.normal(6.5, 2.0)
        else: # C
            math = np.random.normal(6.0, 2.0)
            lit = np.random.normal(8.0, 1.0)
            eng = np.random.normal(7.5, 1.5)
        
        it = np.random.normal(8.5, 1.0) # Tin học mặc định khá cao
        
        data.append({
            "MSHS": f"HS{i:03d}",
            "Họ và Tên": f"Học sinh {i}",
            "Lớp": student_class,
            "Toán": round(np.clip(math, 0, 10), 1),
            "Văn": round(np.clip(lit, 0, 10), 1),
            "Anh": round(np.clip(eng, 0, 10), 1),
            "Tin học": round(np.clip(it, 0, 10), 1)
        })
    df = pd.DataFrame(data)
    df["ĐTB"] = round((df["Toán"] + df["Văn"] + df["Anh"]*2 + df["Tin học"]) / 5, 2)
    return df

@st.cache_data
def load_data_from_repo(filename):
    """
    Hàm này sẽ cố gắng đọc file từ thư mục 'data/' trong repo.
    Nếu không thấy file (do chưa upload), nó sẽ sinh dữ liệu giả lập để demo.
    """
    file_path = os.path.join("data", filename) # Giả sử file nằm trong thư mục data
    
    if os.path.exists(file_path):
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            df.columns = df.columns.str.strip() # Chuẩn hóa tên cột
            return df, "Dữ liệu Thực tế (Github)"
        except Exception as e:
            return None, f"Lỗi đọc file: {str(e)}"
    else:
        # Fallback: Sinh dữ liệu mẫu nếu chưa có file thật
        return generate_mock_data(filename), "Dữ liệu Demo (Chưa tìm thấy file nguồn)"

# Hàm AI Phân tích (Rule-based)
def ai_analyze(df):
    insights = []
    
    # 1. Phân tích môn yếu toàn trường
    subjects = ["Toán", "Văn", "Anh", "Tin học"]
    avg_subjects = df[subjects].mean()
    weakest_subject = avg_subjects.idxmin()
    if avg_subjects[weakest_subject] < 6.5:
        insights.append(f"⚠️ **Cảnh báo môn học:** Môn **{weakest_subject}** có điểm trung bình toàn trường thấp nhất ({avg_subjects[weakest_subject]:.2f}). Cần xem xét lại phương pháp dạy hoặc đề thi.")
    
    # 2. Phân tích độ lệch lớp
    class_avg = df.groupby("Lớp")["ĐTB"].mean()
    best_class = class_avg.idxmax()
    worst_class = class_avg.idxmin()
    diff = class_avg[best_class] - class_avg[worst_class]
    if diff > 2.0:
        insights.append(f"📉 **Chênh lệch chất lượng:** Có sự chênh lệch lớn ({diff:.1f} điểm) giữa lớp dẫn đầu ({best_class}) và lớp cuối bảng ({worst_class}). Cần kế hoạch phụ đạo cho **{worst_class}**.")

    # 3. Phân tích học sinh giỏi/yếu
    top_students = len(df[df["ĐTB"] >= 8.0])
    risk_students = len(df[df["ĐTB"] < 5.0])
    ratio = top_students / (risk_students + 1) # +1 tránh chia cho 0
    if ratio < 1:
        insights.append(f"🚨 **Báo động:** Số lượng học sinh Yếu ({risk_students}) đang nhiều hơn học sinh Giỏi ({top_students}).")
    else:
        insights.append(f"✅ **Tín hiệu tốt:** Tỷ lệ học sinh Giỏi/Yếu đạt mức tích cực ({ratio:.1f}).")
        
    return insights

# --- 4. GIAO DIỆN CHÍNH ---

# Header
col_logo, col_header = st.columns([1, 8])
with col_logo:
    st.image("https://img.icons8.com/color/96/000000/school.png", width=80)
with col_header:
    st.title("HỆ THỐNG QUẢN TRỊ CHẤT LƯỢNG GIÁO DỤC")
    st.markdown("**Trường THPT Phan Bội Châu - Phan Thiết** | *Dành cho Ban Giám Hiệu*")

st.divider()

# --- SIDEBAR: KHU VỰC CHỌN DỮ LIỆU ---
with st.sidebar:
    st.header("🗄️ Kho Dữ Liệu Số")
    st.caption("Dữ liệu được chuẩn hóa bởi Tổ Công nghệ số.")
    
    # Danh sách các file dữ liệu có sẵn (Tổ CN sẽ cập nhật list này)
    available_files = {
        "Học kỳ I - 2025 (Demo)": "kq_hk1_2025.csv",
        "Giữa kỳ I - 2025 (Demo)": "kq_gk1_2025.csv",
        "Khảo sát chất lượng đầu năm": "kscl_2025.csv"
    }
    
    selected_dataset_name = st.selectbox("Chọn kỳ báo cáo:", list(available_files.keys()))
    selected_filename = available_files[selected_dataset_name]
    
    # Load data
    df, status_msg = load_data_from_repo(selected_filename)
    
    if "Demo" in status_msg:
        st.warning(f"⚠️ {status_msg}")
        st.info("💡 Ghi chú cho Tổ CN: Hãy upload file CSV vào thư mục `data/` trên GitHub để thay thế dữ liệu này.")
    else:
        st.success(f"✅ {status_msg}")

# --- XỬ LÝ & HIỂN THỊ MAIN DASHBOARD ---
if df is not None:
    # Sidebar Filters
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔍 Bộ Lọc Hiển Thị")
        all_classes = sorted(df["Lớp"].unique().astype(str))
        selected_class = st.multiselect("Lọc theo Lớp:", all_classes, default=all_classes)
        
    df_filtered = df[df["Lớp"].isin(selected_class)]
    
    # 1. METRICS (KPIs)
    col1, col2, col3, col4 = st.columns(4)
    avg_score = df_filtered["ĐTB"].mean()
    gioi_count = df_filtered[df_filtered["ĐTB"] >= 8.0].shape[0]
    yeu_count = df_filtered[df_filtered["ĐTB"] < 5.0].shape[0]
    
    col1.metric("Tổng số học sinh", f"{len(df_filtered)}", delta="Học sinh")
    col2.metric("Điểm TB Toàn trường", f"{avg_score:.2f}", delta=f"{avg_score - 5.0:.2f} so với chuẩn")
    col3.metric("Học sinh Giỏi", f"{gioi_count}", delta=f"{(gioi_count/len(df_filtered))*100:.1f}%")
    col4.metric("Học sinh Cần lưu ý", f"{yeu_count}", delta=f"-{yeu_count}", delta_color="inverse")

    # 2. PHÂN TÍCH CHI TIẾT (TABS)
    st.markdown("### 📊 Phân Tích Chuyên Sâu")
    tab1, tab2, tab3 = st.tabs(["🤖 Trợ lý Khuyến nghị (AI)", "📈 Biểu đồ Tương quan", "📋 Bảng điểm Chi tiết"])
    
    # TAB 1: AI RECOMMENDATIONS (TÍNH NĂNG MỚI THEO YÊU CẦU 3)
    with tab1:
        st.info("Trợ lý ảo tự động phân tích các mẫu dữ liệu để đưa ra cảnh báo cho Ban Giám hiệu.")
        
        col_ai_1, col_ai_2 = st.columns([2, 1])
        with col_ai_1:
            insights = ai_analyze(df_filtered)
            for insight in insights:
                st.markdown(insight)
            
            if len(insights) == 0:
                st.success("Tuyệt vời! Dữ liệu cho thấy chất lượng giáo dục đang ổn định, chưa phát hiện bất thường lớn.")

        with col_ai_2:
            # Biểu đồ radar so sánh các môn (Chỉ vẽ nếu đủ môn)
            subjects = ["Toán", "Văn", "Anh", "Tin học"]
            avg_radar = df_filtered[subjects].mean().reset_index()
            avg_radar.columns = ['Môn', 'Điểm TB']
            fig_radar = px.line_polar(avg_radar, r='Điểm TB', theta='Môn', line_close=True, title="Biểu đồ năng lực chung")
            fig_radar.update_traces(fill='toself')
            st.plotly_chart(fig_radar, use_container_width=True)

    # TAB 2: TƯƠNG QUAN (CẢI TIẾN THEO YÊU CẦU 1)
    with tab2:
        col_select_1, col_select_2 = st.columns(2)
        with col_select_1:
            x_axis = st.selectbox("Chọn môn đối chiếu (Trục Hoành - X):", ["Toán", "Văn", "Anh", "Tin học", "ĐTB"], index=0)
        with col_select_2:
            y_axis = st.selectbox("Chọn môn so sánh (Trục Tung - Y):", ["Toán", "Văn", "Anh", "Tin học", "ĐTB"], index=2)
            
        col_chart, col_stat = st.columns([3, 1])
        with col_chart:
            fig_corr = px.scatter(
                df_filtered, x=x_axis, y=y_axis, 
                color="Lớp", size="ĐTB", 
                hover_data=["Họ và Tên"],
                title=f"Tương quan giữa {x_axis} và {y_axis}",
                trendline="ols" # Thêm đường xu hướng
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        with col_stat:
            st.markdown("#### Giải thích:")
            st.write(f"- Mỗi chấm là một học sinh.")
            st.write(f"- **Đường thẳng:** Xu hướng chung của mối quan hệ.")
            st.write("- Nếu các chấm phân bố dốc lên: Học tốt môn X thường tốt môn Y.")

    # TAB 3: CHI TIẾT
    with tab3:
        # Tô màu bảng (Try-catch để tránh lỗi)
        try:
            st.dataframe(df_filtered.style.background_gradient(subset=["ĐTB", "Toán", "Văn", "Anh"], cmap="RdYlGn"), use_container_width=True)
        except:
            st.dataframe(df_filtered, use_container_width=True)

else:
    st.error("Không tải được dữ liệu.")
