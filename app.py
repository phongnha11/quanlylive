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

def identify_subjects(df):
    """
    Tự động lọc ra các cột là môn học.
    """
    excluded_cols = ['MSHS', 'Họ và Tên', 'Lớp', 'ĐTB', 'STT', 'Stt', 'Ghi chú']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    subjects = [col for col in numeric_cols if col not in excluded_cols]
    return subjects

def generate_mock_data(semester_name="Học kỳ I"):
    """
    Dữ liệu giả lập chỉ dùng khi KHÔNG CÓ file thật.
    Vẫn cần danh sách lớp giả định để tạo demo.
    """
    classes = ['10A1', '10A2', '10A3', '11B1', '11B2', '12C1', '12C2']
    data = []
    np.random.seed(42 if semester_name == "Học kỳ I" else 24)
    
    for i in range(300):
        student_class = np.random.choice(classes)
        if 'A' in student_class:
            math, lit, eng = np.random.normal(8.0, 1.5), np.random.normal(6.5, 1.5), np.random.normal(7.0, 2.0)
        else:
            math, lit, eng = np.random.normal(6.0, 2.0), np.random.normal(7.5, 1.5), np.random.normal(6.5, 2.0)
        
        # Thêm môn Sinh & Sử demo
        bio, hist = np.random.normal(7.5, 1.5), np.random.normal(6.0, 2.0)
        it = np.random.normal(8.5, 1.0)
        
        data.append({
            "MSHS": f"HS{i:03d}",
            "Họ và Tên": f"Học sinh {i}",
            "Lớp": student_class,
            "Toán": round(np.clip(math, 0, 10), 1),
            "Văn": round(np.clip(lit, 0, 10), 1),
            "Anh": round(np.clip(eng, 0, 10), 1),
            "Tin học": round(np.clip(it, 0, 10), 1),
            "Sinh": round(np.clip(bio, 0, 10), 1),
            "Sử": round(np.clip(hist, 0, 10), 1)
        })
    df = pd.DataFrame(data)
    subject_cols = [c for c in df.columns if c not in ['MSHS', 'Họ và Tên', 'Lớp']]
    df["ĐTB"] = round(df[subject_cols].mean(axis=1), 2)
    return df

@st.cache_data
def load_data_from_repo(filename):
    file_path = os.path.join("data", filename)
    if os.path.exists(file_path):
        try:
            if filename.endswith('.csv'):
                encodings_to_try = ['utf-8', 'utf-8-sig', 'utf-16', 'windows-1258', 'latin1']
                df = None
                for encoding in encodings_to_try:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break 
                    except UnicodeDecodeError:
                        continue
                if df is None: return None, "Lỗi font chữ CSV."
            else:
                df = pd.read_excel(file_path)
            
            df.columns = df.columns.str.strip() 
            return df, "Dữ liệu Thực tế (Github)"
        except Exception as e:
            return None, f"Lỗi đọc file: {str(e)}"
    else:
        return generate_mock_data(filename), "Dữ liệu Demo (Chưa tìm thấy file nguồn)"

def ai_analyze(df):
    insights = []
    subjects = identify_subjects(df)
    
    if subjects:
        avg_subjects = df[subjects].mean()
        weakest_subject = avg_subjects.idxmin()
        if avg_subjects[weakest_subject] < 6.5:
            insights.append(f"⚠️ **Cảnh báo môn học:** Môn **{weakest_subject}** có điểm trung bình thấp nhất ({avg_subjects[weakest_subject]:.2f}).")
    
    if "Lớp" in df.columns and "ĐTB" in df.columns:
        class_avg = df.groupby("Lớp")["ĐTB"].mean()
        best_class = class_avg.idxmax()
        worst_class = class_avg.idxmin()
        diff = class_avg[best_class] - class_avg[worst_class]
        if diff > 2.0:
            insights.append(f"📉 **Chênh lệch:** Có sự chênh lệch lớn ({diff:.1f} điểm) giữa {best_class} và {worst_class}.")

    if "ĐTB" in df.columns:
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
    
    # Cập nhật danh sách file có sẵn trong thư mục data/
    available_files = {
        "Khảo sát chất lượng 2025": "kscl_2025.csv",
        "Học kỳ I (Demo)": "kq_hk1_2025.csv"
    }
    selected_dataset_name = st.selectbox("Chọn kỳ báo cáo:", list(available_files.keys()))
    selected_filename = available_files[selected_dataset_name]
    
    df, status_msg = load_data_from_repo(selected_filename)
    if "Demo" in status_msg: st.warning(f"⚠️ {status_msg}")
    else: st.success(f"✅ {status_msg}")

if df is not None:
    # --- XỬ LÝ LỌC LỚP ĐỘNG ---
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔍 Bộ Lọc Hiển Thị")
        
        # Kiểm tra xem cột 'Lớp' có tồn tại trong file tải lên không
        if "Lớp" in df.columns:
            # Lấy danh sách lớp DUY NHẤT từ dữ liệu hiện tại
            all_classes = sorted(df["Lớp"].unique().astype(str))
            
            # Key quan trọng: Khi tên file thay đổi, widget này sẽ reset
            selected_class = st.multiselect(
                "Lọc theo Lớp:", 
                all_classes, 
                default=all_classes,
                key=f"class_filter_{selected_filename}" 
            )
        else:
            st.error("File dữ liệu thiếu cột 'Lớp'. Vui lòng kiểm tra lại.")
            selected_class = []
        
    # Lọc DataFrame theo lựa chọn
    if "Lớp" in df.columns and selected_class:
        df_filtered = df[df["Lớp"].isin(selected_class)]
    else:
        df_filtered = df # Nếu không có cột Lớp hoặc chưa chọn gì thì hiện tất cả
    
    # --- TỰ ĐỘNG PHÁT HIỆN MÔN ---
    detected_subjects = identify_subjects(df_filtered)
    
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
            st.info(f"Hệ thống đã tự động nhận diện {len(detected_subjects)} môn học.")
            insights = ai_analyze(df_filtered)
            for insight in insights:
                st.markdown(insight)

        with col_ai_2:
            if detected_subjects:
                avg_radar = df_filtered[detected_subjects].mean().reset_index()
                avg_radar.columns = ['Môn', 'Điểm TB']
                fig_radar = px.line_polar(avg_radar, r='Điểm TB', theta='Môn', line_close=True, title="Năng lực chung")
                fig_radar.update_traces(fill='toself')
                st.plotly_chart(fig_radar, use_container_width=True)

    with tab2:
        col_select_1, col_select_2 = st.columns(2)
        cols_for_chart = detected_subjects + ["ĐTB"]
        
        with col_select_1:
            x_axis = st.selectbox("Trục X:", cols_for_chart, index=0 if len(cols_for_chart)>0 else 0)
        with col_select_2:
            default_idx = len(cols_for_chart)-1 if len(cols_for_chart) > 1 else 0
            y_axis = st.selectbox("Trục Y:", cols_for_chart, index=default_idx)
            
        try:
            fig_corr = px.scatter(
                df_filtered, x=x_axis, y=y_axis, 
                color="Lớp" if "Lớp" in df.columns else None, 
                size="ĐTB", 
                title=f"Tương quan {x_axis} - {y_axis}",
                trendline="ols" 
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        except:
            fig_corr = px.scatter(
                df_filtered, x=x_axis, y=y_axis, 
                color="Lớp" if "Lớp" in df.columns else None, 
                size="ĐTB", 
                title=f"Tương quan {x_axis} - {y_axis}"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        try:
            cols_to_color = ["ĐTB"] + detected_subjects
            cols_to_color = [c for c in cols_to_color if c in df_filtered.columns]
            st.dataframe(df_filtered.style.background_gradient(subset=cols_to_color, cmap="RdYlGn"), use_container_width=True)
        except:
            st.dataframe(df_filtered, use_container_width=True)

else:
    st.error("Không tải được dữ liệu.")
