import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import re

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="PBC Digital Hub - Báo Cáo Lãnh Đạo",
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
        border-left: 6px solid #b71c1c; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    h1, h2, h3 { color: #b71c1c; font-family: 'Segoe UI', sans-serif; font-weight: 600; }
    /* Style cho Expander */
    .streamlit-expanderHeader {
        background-color: #ffebee;
        color: #b71c1c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. HÀM XỬ LÝ DỮ LIỆU ---

def identify_subjects(df):
    """Tự động lọc ra các cột là môn học."""
    excluded_cols = ['MSHS', 'Họ và Tên', 'Lớp', 'ĐTB', 'STT', 'Stt', 'Ghi chú', 'Tiến bộ']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    subjects = [col for col in numeric_cols if col not in excluded_cols]
    return subjects

def scan_data_folder():
    """
    Tự động quét thư mục 'data/' để tìm file.
    Yêu cầu tên file định dạng: '01_TênKỳThi.csv' hoặc '02_TênKỳThi.xlsx'
    """
    data_folder = "data"
    files_map = {}
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder) # Tạo thư mục nếu chưa có
        
    # Lấy tất cả file trong thư mục
    files = [f for f in os.listdir(data_folder) if f.endswith(('.csv', '.xlsx', '.xls'))]
    
    # Sắp xếp file theo tên (để 01 luôn đứng trước 02)
    files.sort()
    
    for f in files:
        # Xử lý tên hiển thị đẹp hơn
        # Loại bỏ phần mở rộng
        name_no_ext = os.path.splitext(f)[0]
        # Thay thế dấu gạch dưới bằng khoảng trắng
        display_name = name_no_ext.replace('_', ' ').title()
        files_map[display_name] = f
        
    return files_map

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
                # Xử lý file Excel (xlsx)
                df = pd.read_excel(file_path)
            
            df.columns = df.columns.str.strip() 
            return df, "Dữ liệu Thực tế"
        except Exception as e:
            return None, f"Lỗi đọc file: {str(e)}"
    else:
        return None, "Không tìm thấy file."

def calculate_progress(current_df, current_filename, all_files_map):
    """
    Tính toán sự tiến bộ so với kỳ thi TRƯỚC ĐÓ.
    Dựa vào số thứ tự đầu file (ví dụ: đang xem 02_... thì so sánh với 01_...)
    """
    try:
        # 1. Xác định file trước đó
        # Lấy số thứ tự của file hiện tại (ví dụ: '02_GK1.csv' -> 2)
        current_prefix = int(re.split(r'_|\s', current_filename)[0])
        previous_prefix = current_prefix - 1
        
        previous_file = None
        for display, fname in all_files_map.items():
            if fname.startswith(f"{previous_prefix:02d}") or fname.startswith(f"{previous_prefix}_"):
                previous_file = fname
                break
        
        if not previous_file:
            return current_df, None # Không có kỳ trước để so sánh
            
        # 2. Load dữ liệu cũ
        prev_df, _ = load_data_from_repo(previous_file)
        
        if prev_df is None or "MSHS" not in prev_df.columns or "ĐTB" not in prev_df.columns:
            return current_df, None

        # 3. Merge dữ liệu để tính delta
        # Chỉ lấy cột ĐTB của kỳ trước
        prev_scores = prev_df[["MSHS", "ĐTB"]].rename(columns={"ĐTB": "ĐTB_Cu"})
        merged_df = pd.merge(current_df, prev_scores, on="MSHS", how="left")
        
        # Tính tiến bộ
        merged_df["Tiến bộ"] = merged_df["ĐTB"] - merged_df["ĐTB_Cu"]
        
        return merged_df, previous_file
        
    except Exception as e:
        # Nếu tên file không đúng định dạng số (ví dụ 'Test.csv'), bỏ qua tính năng này
        return current_df, None

def ai_analyze(df):
    insights = []
    subjects = identify_subjects(df)
    
    # 1. Phân tích môn học
    if subjects:
        avg_subjects = df[subjects].mean()
        weakest_subject = avg_subjects.idxmin()
        if avg_subjects[weakest_subject] < 6.5:
            insights.append(f"⚠️ **Cảnh báo môn học:** Môn **{weakest_subject}** có điểm trung bình thấp nhất ({avg_subjects[weakest_subject]:.2f}).")
    
    # 2. Phân tích tiến bộ (Nếu có)
    if "Tiến bộ" in df.columns:
        improved_count = len(df[df["Tiến bộ"] > 0])
        regressed_count = len(df[df["Tiến bộ"] < 0])
        if improved_count > regressed_count:
            insights.append(f"📈 **Xu hướng tích cực:** Có {improved_count} học sinh tiến bộ so với kỳ trước (nhiều hơn số sụt giảm).")
        else:
            insights.append(f"📉 **Xu hướng tiêu cực:** Có {regressed_count} học sinh bị tụt điểm so với kỳ trước. Cần rà soát lại.")

    # 3. Báo động HS Yếu
    if "ĐTB" in df.columns:
        top_students = len(df[df["ĐTB"] >= 8.0])
        risk_students = len(df[df["ĐTB"] < 5.0])
        if risk_students > top_students:
            insights.append(f"🚨 **Báo động:** Số HS Yếu ({risk_students}) đang nhiều hơn HS Giỏi ({top_students}).")
        
    return insights

# --- 4. GIAO DIỆN CHÍNH ---

col_logo, col_header = st.columns([1, 8])
with col_logo:
    st.image("https://img.icons8.com/color/96/000000/school.png", width=80)
with col_header:
    st.title("HỆ THỐNG QUẢN TRỊ CHẤT LƯỢNG GIÁO DỤC")
    st.markdown("**Trường THPT Phan Bội Châu - Phan Thiết** | *Dành cho Ban Giám Hiệu*")

st.divider()

# --- SIDEBAR: KHO DỮ LIỆU TỰ ĐỘNG ---
with st.sidebar:
    st.header("🗄️ Kho Dữ Liệu Số")
    
    # 1. Tự động quét file
    available_files = scan_data_folder()
    
    if not available_files:
        st.error("Chưa có file dữ liệu nào trong thư mục 'data/'.")
        st.info("Vui lòng upload file CSV/XLSX vào GitHub với định dạng: '01_TenKyThi.csv'")
        st.stop()
    
    selected_dataset_name = st.selectbox("Chọn kỳ báo cáo:", list(available_files.keys()))
    selected_filename = available_files[selected_dataset_name]
    
    # Load data cơ bản
    df, status_msg = load_data_from_repo(selected_filename)
    
    if df is not None:
        st.success(f"✅ Đã tải: {selected_filename}")
        
        # 2. Tính toán sự tiến bộ (NẾU CÓ)
        df, prev_file_name = calculate_progress(df, selected_filename, available_files)
        if prev_file_name:
            st.info(f"📊 Đang so sánh với: {prev_file_name}")
    else:
        st.error(status_msg)
        st.stop()

# --- MAIN DASHBOARD ---
if df is not None:
    # --- BỘ LỌC LỚP ---
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔍 Bộ Lọc Hiển Thị")
        if "Lớp" in df.columns:
            all_classes = sorted(df["Lớp"].unique().astype(str))
            selected_class = st.multiselect(
                "Lọc theo Lớp:", all_classes, default=all_classes,
                key=f"class_filter_{selected_filename}" 
            )
            df_filtered = df[df["Lớp"].isin(selected_class)]
        else:
            df_filtered = df

    detected_subjects = identify_subjects(df_filtered)
    
    # --- 1. KPIS & TIẾN BỘ ---
    col1, col2, col3, col4 = st.columns(4)
    avg_score = df_filtered["ĐTB"].mean()
    gioi_count = df_filtered[df_filtered["ĐTB"] >= 8.0].shape[0]
    yeu_count = df_filtered[df_filtered["ĐTB"] < 5.0].shape[0]
    
    # Tính delta tiến bộ trung bình (nếu có)
    delta_progress = None
    if "Tiến bộ" in df_filtered.columns:
        avg_progress = df_filtered["Tiến bộ"].mean()
        delta_progress = f"{avg_progress:+.2f} điểm so với kỳ trước"
    
    col1.metric("Tổng số học sinh", f"{len(df_filtered)}")
    col2.metric("Điểm TB Toàn trường", f"{avg_score:.2f}", delta=delta_progress)
    col3.metric("HS Giỏi", f"{gioi_count}", delta=f"{(gioi_count/len(df_filtered))*100:.1f}%")
    col4.metric("HS Cần lưu ý", f"{yeu_count}", delta=f"-{yeu_count}", delta_color="inverse")

    # --- 2. CHI TIẾT DANH SÁCH CẦN LƯU Ý (Mới) ---
    # Chỉ hiện expander nếu có học sinh yếu
    if yeu_count > 0:
        with st.expander(f"🚨 Bấm để xem danh sách {yeu_count} học sinh Cần lưu ý (ĐTB < 5.0)", expanded=False):
            risk_df = df_filtered[df_filtered["ĐTB"] < 5.0].copy()
            # Chọn các cột quan trọng để hiển thị
            cols_to_show = ["MSHS", "Họ và Tên", "Lớp", "ĐTB"]
            if "Tiến bộ" in risk_df.columns:
                cols_to_show.append("Tiến bộ")
            # Thêm các cột điểm thành phần (nếu có)
            cols_to_show.extend([c for c in detected_subjects if c in risk_df.columns])
            
            st.dataframe(
                risk_df[cols_to_show].sort_values("ĐTB"),
                use_container_width=True,
                hide_index=True
            )

    # --- 3. PHÂN TÍCH CHUYÊN SÂU ---
    st.markdown("### 📊 Phân Tích Chuyên Sâu")
    tab1, tab2, tab3 = st.tabs(["🤖 Trợ lý AI & Xu hướng", "📈 Biểu đồ Tương quan", "📋 Bảng điểm Chi tiết"])
    
    with tab1:
        col_ai_1, col_ai_2 = st.columns([2, 1])
        with col_ai_1:
            st.info(f"Hệ thống đã tự động nhận diện {len(detected_subjects)} môn học.")
            insights = ai_analyze(df_filtered)
            for insight in insights:
                st.markdown(insight)
            
            # Biểu đồ phân bố sự tiến bộ (Histogram) - Nếu có dữ liệu tiến bộ
            if "Tiến bộ" in df_filtered.columns:
                st.markdown("#### 📉 Phân bố sự tiến bộ của học sinh")
                fig_prog = px.histogram(
                    df_filtered, x="Tiến bộ", color="Lớp", 
                    nbins=20, title="Học sinh tiến bộ (Dương) vs Tụt lùi (Âm)"
                )
                st.plotly_chart(fig_prog, use_container_width=True)

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
        if "Tiến bộ" in df_filtered.columns: cols_for_chart.append("Tiến bộ")

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
            if "Tiến bộ" in df_filtered.columns: cols_to_color.append("Tiến bộ")
            
            cols_to_color = [c for c in cols_to_color if c in df_filtered.columns]
            st.dataframe(df_filtered.style.background_gradient(subset=cols_to_color, cmap="RdYlGn"), use_container_width=True)
        except:
            st.dataframe(df_filtered, use_container_width=True)
