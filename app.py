import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- 1. CẤU HÌNH TRANG (Phải đặt đầu tiên) ---
st.set_page_config(
    page_title="PBC Dashboard - Phân Tích Điểm",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS TÙY CHỈNH (Giao diện trường học) ---
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stMetric {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        border-left: 5px solid #003366;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #003366;
        font-family: 'Segoe UI', sans-serif;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. HÀM XỬ LÝ DỮ LIỆU ---

# Hàm tạo dữ liệu giả lập (để demo khi chưa upload file)
@st.cache_data # Cache giúp app chạy nhanh hơn trên Cloud
def generate_mock_data():
    classes = ['10A1', '10A2', '10A3', '11B1', '11B2', '12C1', '12C2']
    data = []
    np.random.seed(42) # Giữ cố định random để demo ổn định
    
    for i in range(500):
        student_class = np.random.choice(classes)
        # Logic: Lớp A giỏi Toán, Lớp C giỏi Văn
        if 'A' in student_class:
            math = np.random.normal(8.0, 1.5)
            lit = np.random.normal(6.5, 1.5)
        else:
            math = np.random.normal(6.0, 2.0)
            lit = np.random.normal(7.5, 1.5)
            
        eng = np.random.normal(7.0, 2.0)
        it = np.random.normal(8.5, 1.0)
        
        # Clip điểm 0-10
        math = np.clip(math, 0, 10)
        lit = np.clip(lit, 0, 10)
        eng = np.clip(eng, 0, 10)
        it = np.clip(it, 0, 10)
        
        data.append({
            "MSHS": f"HS{i:03d}",
            "Họ và Tên": f"Học sinh {i}",
            "Lớp": student_class,
            "Toán": round(math, 1),
            "Văn": round(lit, 1),
            "Anh": round(eng, 1),
            "Tin học": round(it, 1)
        })
    
    df = pd.DataFrame(data)
    df["ĐTB"] = round((df["Toán"] + df["Văn"] + df["Anh"]*2 + df["Tin học"]) / 5, 2)
    return df

# Hàm tải dữ liệu từ file Excel tải lên
@st.cache_data
def load_data(uploaded_file):
    try:
        # Hỗ trợ cả CSV và Excel
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return None

# --- 4. GIAO DIỆN CHÍNH ---

# Header
col_logo, col_header = st.columns([1, 6])
with col_logo:
    # Bạn có thể thay link ảnh logo trường ở đây
    st.image("https://img.icons8.com/color/96/000000/school.png", width=70)
with col_header:
    st.title("THPT PHAN BỘI CHÂU - DIGITAL HUB")
    st.caption("Hệ thống quản trị chất lượng giáo dục dựa trên dữ liệu (Data-Driven Education)")

st.divider()

# Sidebar: Công cụ điều khiển
with st.sidebar:
    st.header("⚙️ Bảng Điều Khiển")
    
    # Upload File
    uploaded_file = st.file_uploader("📂 Tải lên bảng điểm (Excel/CSV)", type=["xlsx", "csv", "xls"])
    
    st.info("💡 **Mẹo:** Nếu chưa có file, hệ thống sẽ chạy dữ liệu mẫu mô phỏng.")
    
    # Nút tải file mẫu (để GV biết định dạng nhập)
    sample_csv = generate_mock_data().to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Tải file mẫu định dạng chuẩn",
        data=sample_csv,
        file_name='mau_nhap_diem_pbc.csv',
        mime='text/csv',
    )

# Xử lý dữ liệu đầu vào
if uploaded_file is not None:
    df = load_data(uploaded_file)
    data_source = "Dữ liệu thực tế từ File"
else:
    df = generate_mock_data()
    data_source = "Dữ liệu Giả lập (Demo)"

if df is not None:
    # Sidebar Filters (Sau khi có dữ liệu mới hiện bộ lọc)
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔍 Bộ Lọc")
        all_classes = sorted(df["Lớp"].unique())
        selected_class = st.multiselect("Chọn Lớp:", all_classes, default=all_classes)
    
    # Filter DataFrame
    df_filtered = df[df["Lớp"].isin(selected_class)]
    
    # --- DASHBOARD CONTENT ---
    
    # Row 1: Thông báo nguồn dữ liệu
    if uploaded_file is None:
        st.warning(f"⚠️ Đang hiển thị: **{data_source}**. Hãy tải file lên để xem kết quả thực.", icon="🖥️")
    else:
        st.success(f"✅ Đang hiển thị: **{data_source}**. Tổng số: {len(df)} học sinh.", icon="📂")

    # Row 2: KPIs
    col1, col2, col3, col4 = st.columns(4)
    avg_score = df_filtered["ĐTB"].mean()
    gioi_count = df_filtered[df_filtered["ĐTB"] >= 8.0].shape[0]
    yeu_count = df_filtered[df_filtered["ĐTB"] < 5.0].shape[0]
    
    col1.metric("Sĩ số đang xem", f"{len(df_filtered)} em")
    col2.metric("Điểm TB toàn trường", f"{avg_score:.2f}")
    col3.metric("Học sinh Giỏi (>8.0)", f"{gioi_count} em", delta=f"{gioi_count/len(df_filtered)*100:.1f}%")
    col4.metric("Cần Cải thiện (<5.0)", f"{yeu_count} em", delta=f"-{yeu_count}", delta_color="inverse")

    # Row 3: Biểu đồ
    tab1, tab2, tab3 = st.tabs(["📊 Phổ Điểm", "📉 Tương Quan Môn Học", "📋 Danh Sách Chi Tiết"])
    
    with tab1:
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            subject = st.selectbox("Chọn môn phân tích:", ["Toán", "Văn", "Anh", "Tin học", "ĐTB"])
            fig = px.histogram(df_filtered, x=subject, color="Lớp", nbins=15, 
                               title=f"Phổ điểm môn {subject}", barmode="overlay", opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)
        with col_c2:
            st.markdown(f"**Nhận xét nhanh môn {subject}:**")
            max_score = df_filtered[subject].max()
            min_score = df_filtered[subject].min()
            st.write(f"- Cao nhất: **{max_score}**")
            st.write(f"- Thấp nhất: **{min_score}**")
            st.progress((df_filtered[subject].mean()/10), text=f"Trung bình: {df_filtered[subject].mean():.1f}/10")

    with tab2:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_scat = px.scatter(df_filtered, x="Toán", y="Tin học", color="Lớp", size="ĐTB", 
                                  title="Tương quan năng lực Tư duy logic (Toán vs Tin)")
            st.plotly_chart(fig_scat, use_container_width=True)
        with col_s2:
            fig_scat2 = px.scatter(df_filtered, x="Văn", y="Anh", color="Lớp", size="ĐTB", 
                                   title="Tương quan năng lực Ngôn ngữ (Văn vs Anh)")
            st.plotly_chart(fig_scat2, use_container_width=True)

    with tab3:
        st.dataframe(df_filtered.style.background_gradient(subset=["ĐTB"], cmap="RdYlGn"), use_container_width=True)

else:
    st.error("File tải lên không đúng định dạng. Vui lòng tải file mẫu và thử lại.")