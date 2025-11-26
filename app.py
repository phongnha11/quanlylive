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

# Hàm tạo dữ liệu giả lập (Fallback)
def generate_mock_data(semester_name="Học kỳ I"):
    classes = ['10A1', '10A2', '10A3', '11B1', '11B2', '12C1', '12C2']
    data = []
    np.random.seed(42 if semester_name == "Học kỳ I" else 24)
    
    for i in range(300):
        student_class = np.random.choice(classes)
        if 'A' in student_class:
            math = np.random.normal(8.0, 1.5)
            lit = np.random.normal(6.5, 1.5)
            eng = np.random.normal(7.0, 2.0)
        elif 'B' in student_class:
            math = np.random.normal(7.0, 1.5)
            lit = np.random.normal(7.0, 1.5)
            eng = np.random.normal(6.5, 2.0)
        else:
            math = np.random.normal(6.0, 2.0)
            lit = np.random.normal(8.0, 1.0)
            eng = np.random.normal(7.5, 1.5)
        
        it = np.random.normal(8.5, 1.0)
        
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
    Hàm đọc dữ liệu thông minh: Tự động thử các bảng mã khác nhau 
    để xử lý lỗi font tiếng Việt.
    """
    file_path = os.path.join("data", filename)
    
    if os.path.exists(file_path):
        try:
            if filename.endswith('.csv'):
                # --- NÂNG CẤP: THỬ NHIỀU BẢNG MÃ ---
                # Danh sách các bảng mã phổ biến ở Việt Nam
                encodings_to_try = ['utf-8', 'utf-8-sig', 'utf-16', 'windows-1258', 'latin1']
                
                df = None
                for encoding in encodings_to_try:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break # Nếu đọc thành công thì thoát vòng lặp
                    except UnicodeDecodeError:
                        continue # Nếu lỗi thì thử bảng mã tiếp theo
                
                if df is None:
                    return None, "Lỗi: Không thể đọc được font chữ của file CSV này."
                    
            else:
                df = pd.read_excel(file_path)
            
            df.columns = df.columns.str.strip() # Chuẩn hóa tên cột
            return df, "Dữ liệu Thực tế (Github)"
        except Exception as e:
            return None, f"Lỗi đọc file: {str(e)}"
    else:
        return generate_mock_data(filename), "Dữ liệu Demo (Chưa tìm thấy file nguồn)"

# Hàm AI Phân tích
def ai_analyze(df):
    insights = []
    
    # 1. Môn yếu
    subjects = ["Toán", "Văn", "Anh", "Tin học"]
    # Chỉ lấy các môn có trong file
    available_subjects = [s for s in subjects if s in df.columns]
    
    if available_subjects:
        avg_subjects = df[available_subjects].mean()
        weakest_subject = avg_subjects.idxmin()
        if avg_subjects[weakest_subject] < 6.5:
            insights.append(f"⚠️ **Cảnh báo môn học:** Môn **{weakest_subject}** có điểm trung bình thấp nhất ({avg_subjects[weakest_subject]:.2f}).")
    
    # 2. Độ lệch lớp
    class_avg = df.groupby("Lớp")["ĐTB"].mean()
    best_class = class_avg.idxmax()
    worst_class = class_avg.idxmin()
    diff = class_avg[best_class] - class_avg[worst_class]
    if diff > 2.0:
        insights.append(f"📉 **Chênh lệch:** Có sự chênh lệch lớn ({diff:.1f} điểm) giữa {best_class} và {worst_class}.")

    # 3. Tỷ lệ Giỏi/Yếu
    top_students = len(df[df["ĐTB"] >= 8.0])
    risk_students = len(df[df["ĐTB"] < 5.0])
    ratio = top_students / (risk_students + 1)
    if ratio < 1:
        insights.append(f"🚨 **Báo động:** Số HS Yếu ({risk_students}) nhiều hơn HS Giỏi ({top_students}).")
    else:
        insights.append(f"✅ **Tín hiệu tốt:** Tỷ lệ HS Giỏi cao hơn HS Yếu.")
        
    return insights

# --- 4. GIAO DIỆN CHÍNH ---

col_logo, col_header = st.columns([1, 8])
with col_logo:
    st.image("https://img.icons8.com/color/96/000000/school.png", width=80)
with col_header:
    st.title("HỆ THỐNG QUẢN TRỊ CHẤT LƯỢNG GIÁO DỤC")
    st.markdown("**Trường THPT Phan Bội Châu - Phan Thiết** | *Dành cho Ban Giám Hiệu*")

st.divider()

with st.sidebar:
    st.header("🗄️ Kho Dữ Liệu Số")
    
    # Cập nhật đúng tên file bạn đã upload
    available_files = {
        "Khảo sát chất lượng 2025": "kscl_2025.csv",
        "Học kỳ I (Demo)": "kq_hk1_2025.csv"
    }
    
    selected_dataset_name = st.selectbox("Chọn kỳ báo cáo:", list(available_files.keys()))
    selected_filename = available_files[selected_dataset_name]
    
    df, status_msg = load_data_from_repo(selected_filename)
    
    if "Demo" in status_msg:
        st.warning(f"⚠️ {status_msg}")
    else:
        st.success(f"✅ {status_msg}")

if df is not None:
    # Sidebar Filters
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔍 Bộ Lọc Hiển Thị")
        all_classes = sorted(df["Lớp"].unique().astype(str))
        selected_class = st.multiselect("Lọc theo Lớp:", all_classes, default=all_classes)
        
    df_filtered = df[df["Lớp"].isin(selected_class)]
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    avg_score = df_filtered["ĐTB"].mean()
    gioi_count = df_filtered[df_filtered["ĐTB"] >= 8.0].shape[0]
    yeu_count = df_filtered[df_filtered["ĐTB"] < 5.0].shape[0]
    
    col1.metric("Tổng số học sinh", f"{len(df_filtered)}")
    col2.metric("Điểm TB", f"{avg_score:.2f}")
    col3.metric("HS Giỏi", f"{gioi_count}", delta=f"{(gioi_count/len(df_filtered))*100:.1f}%")
    col4.metric("Cần lưu ý", f"{yeu_count}", delta=f"-{yeu_count}", delta_color="inverse")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🤖 Trợ lý AI", "📈 Biểu đồ Tương quan", "📋 Bảng điểm"])
    
    with tab1:
        col_ai_1, col_ai_2 = st.columns([2, 1])
        with col_ai_1:
            st.info("Trợ lý ảo phân tích tự động:")
            insights = ai_analyze(df_filtered)
            for insight in insights:
                st.markdown(insight)

        with col_ai_2:
            # Radar Chart
            subjects = ["Toán", "Văn", "Anh", "Tin học"]
            available_subjects = [s for s in subjects if s in df_filtered.columns]
            if available_subjects:
                avg_radar = df_filtered[available_subjects].mean().reset_index()
                avg_radar.columns = ['Môn', 'Điểm TB']
                fig_radar = px.line_polar(avg_radar, r='Điểm TB', theta='Môn', line_close=True, title="Năng lực chung")
                fig_radar.update_traces(fill='toself')
                st.plotly_chart(fig_radar, use_container_width=True)

    with tab2:
        col_select_1, col_select_2 = st.columns(2)
        cols_for_chart = [c for c in ["Toán", "Văn", "Anh", "Tin học", "ĐTB"] if c in df_filtered.columns]
        
        with col_select_1:
            x_axis = st.selectbox("Trục X:", cols_for_chart, index=0)
        with col_select_2:
            y_axis = st.selectbox("Trục Y:", cols_for_chart, index=min(2, len(cols_for_chart)-1))
            
        try:
            fig_corr = px.scatter(
                df_filtered, x=x_axis, y=y_axis, 
                color="Lớp", size="ĐTB", 
                title=f"Tương quan {x_axis} - {y_axis}",
                trendline="ols" 
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        except:
            # Fallback nếu lỗi trendline
            fig_corr = px.scatter(
                df_filtered, x=x_axis, y=y_axis, 
                color="Lớp", size="ĐTB", 
                title=f"Tương quan {x_axis} - {y_axis}"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        try:
            # Tô màu chỉ cho các cột tồn tại
            cols_to_color = [c for c in ["ĐTB", "Toán", "Văn", "Anh"] if c in df_filtered.columns]
            st.dataframe(df_filtered.style.background_gradient(subset=cols_to_color, cmap="RdYlGn"), use_container_width=True)
        except:
            st.dataframe(df_filtered, use_container_width=True)
