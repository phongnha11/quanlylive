import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="PBC Dashboard - Phân Tích Điểm",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS TÙY CHỈNH ---
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .stMetric {
        background-color: white; padding: 10px; border-radius: 8px;
        border-left: 5px solid #003366; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1, h2, h3 { color: #003366; font-family: 'Segoe UI', sans-serif; }
</style>
""", unsafe_allow_html=True)

# --- 3. HÀM XỬ LÝ DỮ LIỆU ---

@st.cache_data
def generate_mock_data():
    classes = ['10A1', '10A2', '10A3', '11B1', '11B2', '12C1', '12C2']
    data = []
    np.random.seed(42)
    for i in range(200):
        student_class = np.random.choice(classes)
        if 'A' in student_class:
            math, lit = np.random.normal(8.0, 1.5), np.random.normal(6.5, 1.5)
        else:
            math, lit = np.random.normal(6.0, 2.0), np.random.normal(7.5, 1.5)
        
        data.append({
            "MSHS": f"HS{i:03d}",
            "Họ và Tên": f"Học sinh {i}",
            "Lớp": student_class,
            "Toán": round(np.clip(math, 0, 10), 1),
            "Văn": round(np.clip(lit, 0, 10), 1),
            "Anh": round(np.clip(np.random.normal(7.0, 2.0), 0, 10), 1),
            "Tin học": round(np.clip(np.random.normal(8.5, 1.0), 0, 10), 1)
        })
    df = pd.DataFrame(data)
    df["ĐTB"] = round((df["Toán"] + df["Văn"] + df["Anh"]*2 + df["Tin học"]) / 5, 2)
    return df

@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # --- QUAN TRỌNG: CHUẨN HÓA TÊN CỘT ---
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return None

# --- 4. GIAO DIỆN CHÍNH ---

col_logo, col_header = st.columns([1, 6])
with col_logo:
    st.image("https://img.icons8.com/color/96/000000/school.png", width=70)
with col_header:
    st.title("THPT PHAN BỘI CHÂU - DIGITAL HUB")
    st.caption("Hệ thống quản trị chất lượng giáo dục dựa trên dữ liệu")

st.divider()

with st.sidebar:
    st.header("⚙️ Bảng Điều Khiển")
    uploaded_file = st.file_uploader("📂 Tải lên bảng điểm", type=["xlsx", "csv", "xls"])
    
    st.info("Nếu chưa có file, hệ thống sẽ chạy dữ liệu mẫu.")
    
    # Nút tải file mẫu
    sample_csv = generate_mock_data().to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Tải file mẫu chuẩn", data=sample_csv, file_name='mau_nhap_diem_pbc.csv', mime='text/csv')

# Xử lý dữ liệu đầu vào
if uploaded_file is not None:
    df = load_data(uploaded_file)
    data_source = "Dữ liệu thực tế"
else:
    df = generate_mock_data()
    data_source = "Dữ liệu Giả lập"

# --- KIỂM TRA DỮ LIỆU HỢP LỆ ---
if df is not None:
    required_cols = ["Lớp", "Toán", "Văn", "Anh", "Tin học", "ĐTB"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ **Lỗi File:** Thiếu cột: {', '.join(missing_cols)}")
        st.stop()
    
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔍 Bộ Lọc")
        all_classes = sorted(df["Lớp"].unique().astype(str))
        selected_class = st.multiselect("Chọn Lớp:", all_classes, default=all_classes)
    
    df_filtered = df[df["Lớp"].isin(selected_class)]
    
    if uploaded_file is None:
        st.warning(f"⚠️ Đang hiển thị: **{data_source}**.", icon="🖥️")
    else:
        st.success(f"✅ Đang hiển thị: **{data_source}**. Tổng: {len(df)} HS.", icon="📂")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    avg_score = df_filtered["ĐTB"].mean()
    gioi_count = df_filtered[df_filtered["ĐTB"] >= 8.0].shape[0]
    yeu_count = df_filtered[df_filtered["ĐTB"] < 5.0].shape[0]
    
    col1.metric("Sĩ số", f"{len(df_filtered)}")
    col2.metric("Điểm TB", f"{avg_score:.2f}")
    
    delta_gioi = f"{gioi_count/len(df_filtered)*100:.1f}%" if len(df_filtered) > 0 else "0%"
    col3.metric("Giỏi (>8.0)", f"{gioi_count}", delta=delta_gioi)
    col4.metric("Yếu (<5.0)", f"{yeu_count}", delta=f"-{yeu_count}", delta_color="inverse")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Phổ Điểm", "📉 Tương Quan", "📋 Chi Tiết"])
    
    with tab1:
        col_c1, col_c2 = st.columns([3, 1])
        with col_c1:
            subject = st.selectbox("Chọn môn:", ["Toán", "Văn", "Anh", "Tin học", "ĐTB"])
            fig = px.histogram(df_filtered, x=subject, color="Lớp", nbins=15, barmode="overlay", opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)
        with col_c2:
            st.info(f"Cao nhất: {df_filtered[subject].max()}")
            st.warning(f"Thấp nhất: {df_filtered[subject].min()}")

    with tab2:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.plotly_chart(px.scatter(df_filtered, x="Toán", y="Tin học", color="Lớp", size="ĐTB", title="Toán vs Tin"), use_container_width=True)
        with col_s2:
            st.plotly_chart(px.scatter(df_filtered, x="Văn", y="Anh", color="Lớp", size="ĐTB", title="Văn vs Anh"), use_container_width=True)

    with tab3:
        # --- SỬA LỖI QUAN TRỌNG: Try-Catch cho phần tô màu ---
        try:
            st.dataframe(df_filtered.style.background_gradient(subset=["ĐTB"], cmap="RdYlGn"), use_container_width=True)
        except Exception:
            # Nếu tô màu thất bại (do thiếu thư viện hoặc lỗi khác), hiển thị bảng trơn
            st.warning("⚠️ Chế độ hiển thị đơn giản (Không màu nền) đang được kích hoạt.")
            st.dataframe(df_filtered, use_container_width=True)

else:
    st.error("File tải lên bị lỗi hoặc không đọc được.")
