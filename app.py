import os
import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.figure_factory as ff

from sklearn.ensemble import GradientBoostingClassifier
try:
    from surprise import dump
except ImportError:
    st.error("Thiếu thư viện 'scikit-surprise'. Hãy chạy: pip install scikit-surprise")

# import các hàm tự viết từ utils.py
# from utils import load_model, perform_clustering, get_recommendations, train_models

# --- CONFIG ---
st.set_page_config(page_title="E-Com Analytics", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .sidebar-title {
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        color: #005f6b;
        margin-top: -15px;
        line-height: 1.2;
    }
</style>
""", unsafe_allow_html=True)

## SIDEBAR - LOGO & THÔNG TIN
with st.sidebar:
    # Tạo 3 cột để ép logo vào giữa
    col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])
    with col_logo2:
        # Sử dụng logo UTE theo link bạn cung cấp
        st.image("https://dashboardero.hcmute.edu.vn/static/assets/images/ute_logo.png", use_container_width=True)
    
    # Tên trường sử dụng class CSS đã định nghĩa ở trên
    st.markdown('<p class="sidebar-title">TRƯỜNG ĐH CÔNG NGHỆ KỸ THUẬT TP.HCM</p>', unsafe_allow_html=True)
    
    st.divider()
    
    # Thông tin môn học và nhóm
    st.markdown("### 📘 Thông tin đồ án")
    st.write("**Môn học:** BIG DATA")
    st.write("**Giảng viên:** ThS. Hồ Nhựt Minh")
    st.write("**Thực hiện:** Nhóm 07")

    st.divider()

    # Menu điều hướng (Option Menu) đặt ở dưới cùng của thông tin nhóm
    selected = option_menu(
        menu_title="DANH MỤC", 
        options=["Dashboard", "Phân khúc", "Khuyến nghị", "Xu hướng", "Dự đoán", "Admin"],
        icons=["speedometer2", "people", "gift", "graph-up-arrow", "magic", "gear"], 
        menu_icon="cast", 
        default_index=0,
    )
    
    st.markdown("---")

    # Tạo hộp thoại đóng/mở chứa danh sách file
    with st.expander("📁 Danh mục Dữ liệu (CSV)"):
        st.markdown("""
        Hệ thống cần nhập các file dữ liệu sau:
        - 📄 v` 
        - 📄 `data/data_segmentev`
        - 📄 `data/rfm_clustered.csv`
        """)
        
    st.caption("Developed by Group 07 @ 2026")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    # File cho biểu đồ thời gian và bang
    df_dash = pd.read_csv('data/olist_dashboard_final.csv')
    df_dash['month_year'] = df_dash['month_year'].astype(str)
    
    # File cho phân cụm (Clustering) - File bạn vừa gửi
    df_rfm = pd.read_csv('data/rfm_clustered.csv')
    
    return df_dash, df_rfm
    
# --- LOAD RECOMMENDER ASSETS (SVD) ---
@st.cache_resource
def load_recommender_assets():
    try:
        from surprise import dump
        # Tải mô hình SVD
        _, model = dump.load('models/svd_model.pkl')
        
        # Tải danh sách sản phẩm (Giữ nguyên dạng DataFrame để lát nữa lấy Tên)
        p_df = pd.read_csv('data/product_list')
        return model, p_df
    except Exception as e:
        return None, None
    
@st.cache_resource
def load_trained_model():
    model_path = 'models/trained_model'
    if os.path.exists(model_path): 
        return joblib.load(model_path)
    return None

# Gọi hàm load dữ liệu tổng quát
df, df_rfm = load_data()
df, df_rfm = load_data()

        
# --- MAIN BODY ROUTING ---
if selected == "Dashboard":
    st.header("📊 Tổng quan Thị trường & Phân cụm Khách hàng")
    
    # --- PHẦN 0: CSS ĐỂ TẠO KPI CARDS CÓ MÀU SẮC ---
    st.markdown("""
        <style>
        .kpi-card {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .card-revenue { background-color: #E8F0FE; color: #1967D2; }
        .card-orders { background-color: #E6F4EA; color: #137333; }
        .card-aov { background-color: #FEF7E0; color: #B06000; }
        .kpi-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 10px; color: #555;}
        .kpi-value { font-size: 2rem; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

    # --- PHẦN 1: THỐNG KÊ MÔ TẢ (EDA) ---
    total_revenue = df['payment_value'].sum()
    total_orders = df['order_id'].sum()
    aov = total_revenue / total_orders if total_orders > 0 else 0
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
            <div class="kpi-card card-revenue">
                <div class="kpi-title">Tổng doanh thu</div>
                <div class="kpi-value">{total_revenue/1e6:.2f}M BRL</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="kpi-card card-orders">
                <div class="kpi-title">Tổng đơn hàng</div>
                <div class="kpi-value">{total_orders:,}</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class="kpi-card card-aov">
                <div class="kpi-title">AOV (TB đơn)</div>
                <div class="kpi-value">{aov:.2f} BRL</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📈 Doanh thu theo tháng")
        df_monthly_sum = df.groupby('month_year')['payment_value'].sum().reset_index()
        fig_line = px.line(df_monthly_sum, x='month_year', y='payment_value', markers=True, template="plotly_white")
        st.plotly_chart(fig_line, use_container_width=True)
    with col2:
        st.subheader("📍 Top 5 Bang")
        
        # 1. Định nghĩa Mapping tên đầy đủ của bang
        state_map = {
            'SP': 'São Paulo', 
            'RJ': 'Rio de Janeiro', 
            'MG': 'Minas Gerais', 
            'RS': 'Rio Grande do Sul', 
            'PR': 'Paraná'
        }
        
        # 2. Xử lý dữ liệu
        df_state = df.groupby('customer_state')['payment_value'].sum().nlargest(5).reset_index()
        # Ánh xạ mã bang sang tên đầy đủ, nếu không có trong map thì giữ nguyên mã
        df_state['State Name'] = df_state['customer_state'].map(state_map).fillna(df_state['customer_state'])

        # 3. Vẽ biểu đồ Bar Chart
        fig_bar = px.bar(
            df_state, 
            x='customer_state', 
            y='payment_value', 
            color='State Name', # Sử dụng tên đầy đủ cho phần chú thích (Legend)
            template="plotly_white", 
            labels={'customer_state': 'Mã Bang', 'payment_value': 'Doanh thu (BRL)', 'State Name': 'Tên Bang'}
        )
        
        # Tinh chỉnh chú giải nằm ngang phía dưới để biểu đồ rộng hơn
        fig_bar.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
            margin=dict(t=20, b=20)
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # 4. Nhận xét sơ bộ (Tự động lấy tên bang cao nhất)
        top_1_name = df_state.iloc[0]['State Name']
        st.info(f"**Nhận xét:** Bang **{top_1_name}** dẫn đầu vượt trội về doanh thu. Đây là thị trường trọng điểm cần tập trung các chiến dịch khuyến mãi và tối ưu kho vận.")
   
    # --- PHẦN MỚI: THÊM 4 HÌNH ẢNH BIỂU ĐỒ  ---
    st.markdown("---")
    st.subheader("📊 Phân bố review_score")

    # Tạo 2 hàng, mỗi hàng 2 cột
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        st.write("**Phân bố giá sản phẩm (< 500 BRL)**")
        st.image("product_distribution.png", use_container_width=True)

    with row1_col2:
        st.write("**Phân bố trạng thái đơn hàng**")
        st.image("order_status.png", use_container_width=True)

    with row2_col1:
        st.write("**Phân bố Review Score**")
        st.image("review_score.png", use_container_width=True)

    with row2_col2:
        st.write("**Phân bố giá trị thanh toán (< 1000 BRL)**")
        st.image("Phân bố giá trị thanh toán (nhỏ hơn 1000 BRL).png", use_container_width=True)

    st.markdown("---") # Đường kẻ ngăn cách phần EDA và Clustering    
    
    # --- PHẦN 2: KẾT QUẢ CLUSTERING (Sử dụng file rfm_clustered.csv) ---
    st.subheader("👥 Kết quả Phân cụm Khách hàng (K-Means)")
    
    # Tính toán đặc trưng từng cụm (Thống kê mô tả Cluster)
    cluster_stats = df_rfm.groupby('KMeans_Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'customer_unique_id': 'count'
    }).reset_index()

    tab1, tab2 = st.tabs(["Biểu đồ Phân bổ", "Trực quan hóa 3D"])

    with tab1:
        col_left, col_right = st.columns(2)
        with col_left:
            fig_pie = px.pie(cluster_stats, values='customer_unique_id', names='KMeans_Cluster', 
                             title="Tỷ lệ khách hàng giữa các cụm", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_right:
            fig_box = px.bar(cluster_stats, x='KMeans_Cluster', y='Monetary', 
                             title="Giá trị chi tiêu trung bình mỗi cụm", color='KMeans_Cluster')
            st.plotly_chart(fig_box, use_container_width=True)

    with tab2:
        st.write("**Không gian Phân cụm 3D (R-F-M)**")
        # Vẽ biểu đồ 3D để thể hiện sự tách biệt của các cụm
        fig_3d = px.scatter_3d(df_rfm.sample(2000), # Sample để app chạy mượt hơn
                               x='Recency', y='Frequency', z='Monetary',
                               color='KMeans_Cluster', opacity=0.7,
                               title="Trực quan hóa Phân cụm khách hàng",
                               log_z=True) # Log scale cho tiền để dễ nhìn
        st.plotly_chart(fig_3d, use_container_width=True)

    # Hiển thị bảng giải thích
    with st.expander("💡 Giải thích ý nghĩa các cụm"):
        st.write("""
        - **Cụm có Monetary cao:** Nhóm khách hàng VIP đem lại nhiều doanh thu nhất.
        - **Cụm có Recency cao:** Nhóm khách hàng cũ đã lâu không mua hàng (Cần chăm sóc lại).
        - **Cụm có Recency thấp & Frequency thấp:** Khách hàng mới.
        """)
        st.dataframe(cluster_stats.style.format("{:.2f}"))

elif selected == "Phân khúc":
    st.header("🔍 Phân cụm khách hàng bằng mô hình RFM + KMeans")
    
    import joblib
    import plotly.express as px
    import pandas as pd
    from sklearn.cluster import KMeans # Đã thêm import KMeans để vẽ Elbow

    # --- LOAD MÔ HÌNH ---
    @st.cache_resource
    def load_kmeans_pipeline():
        try:
            return joblib.load('models/kmeans_pipeline.joblib')
        except:
            return None

    kmeans_pipeline = load_kmeans_pipeline()
    tab_overview, tab_predict = st.tabs(["📊 Phân tích tập dữ liệu đã có", "🎯 Dự đoán Nhóm khách hàng mới (AI)"])

    # Bảng màu cho các cụm 
    cluster_colors = {
        "0": "#E59866", # Cam đất nhạt - Nhóm VIP
        "1": "#7FB3D5", # Xanh dương pastel - Nhóm Tiềm năng
        "2": "#D5D8DC", # Xám bạc nhạt - Nhóm Ngủ đông
        "3": "#A9DFBF"  # Xanh lá pastel - Nhóm Mới
    }

    # ==========================================
    # TAB 1: PHÂN TÍCH
    # ==========================================
    with tab_overview:
        uploaded_file = st.file_uploader("Tải lên file dữ liệu đã phân cụm (CSV)", type=["csv"], key="upload_clustered")
        
        if uploaded_file is not None:
            df_seg = pd.read_csv(uploaded_file)
            
            if 'KMeans_Cluster' not in df_seg.columns:
                st.error("⚠️ File không có cột 'KMeans_Cluster'.")
            else:
                # Ép kiểu cột Cluster về string để đồng bộ map màu sắc
                df_seg['KMeans_Cluster'] = df_seg['KMeans_Cluster'].astype(str)

                # --- 1. BIỂU ĐỒ KHUỶU TAY (ELBOW METHOD) ---
                st.subheader("📉 Phương pháp Khuỷu tay (Elbow Method)")
                with st.expander("Bấm để xem biểu đồ đánh giá số cụm tối ưu (K)"):
                    rfm_cols = ['Recency', 'Frequency', 'Monetary']
                    if all(col in df_seg.columns for col in rfm_cols):
                        with st.spinner("Đang tính toán WCSS... (Có thể mất vài giây)"):
                            # Lấy mẫu tối đa 5000 dòng để tính toán Elbow nhanh hơn mà vẫn giữ được độ chính xác
                            sample_df = df_seg[rfm_cols].sample(min(5000, len(df_seg)), random_state=42)
                            wcss = []
                            K_range = range(1, 11)
                            for k in K_range:
                                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                                kmeans.fit(sample_df)
                                wcss.append(kmeans.inertia_)
                            
                            df_elbow = pd.DataFrame({'Số lượng cụm (k)': list(K_range), 'WCSS (Inertia)': wcss})
                            fig_elbow = px.line(df_elbow, x='Số lượng cụm (k)', y='WCSS (Inertia)', markers=True, 
                                                template="plotly_white")
                            fig_elbow.update_traces(line_color="#6C8EBF", marker=dict(size=8, color="#6C8EBF"))
                            
                            # Đánh dấu đường kẻ vạch ngay tại số cụm hiện tại đang dùng (thường là 4)
                            current_k = df_seg['KMeans_Cluster'].nunique()
                            if current_k in K_range:
                                fig_elbow.add_vline(x=current_k, line_dash="dash", line_color="red", 
                                                    annotation_text=f" k = {current_k}", annotation_position="top right")
                            
                            st.plotly_chart(fig_elbow, use_container_width=True)
                            st.caption("*Biểu đồ minh họa sự sụt giảm của tổng bình phương khoảng cách (WCSS). Điểm khuỷu tay (elbow) thường là lựa chọn tối ưu cho K.*")
                    else:
                        st.warning("⚠️ Cần có đủ 3 cột Recency, Frequency, Monetary để vẽ biểu đồ này.")

                st.markdown("---")

                # --- 2. BIỂU ĐỒ TỔNG QUAN ---
                col_left, col_right = st.columns(2)
                with col_left:
                    st.subheader("Số lượng khách hàng mỗi nhóm")
                    count_seg = df_seg['KMeans_Cluster'].value_counts().reset_index()
                    count_seg.columns = ['Nhóm', 'Số lượng']
                    # Sắp xếp để nhóm hiện theo thứ tự 0, 1, 2, 3
                    count_seg = count_seg.sort_values('Nhóm')
                    
                    fig_count = px.bar(count_seg, x='Nhóm', y='Số lượng', color='Nhóm', text_auto=True, 
                                       color_discrete_map=cluster_colors, template="plotly_white")
                    fig_count.update_layout(showlegend=False)
                    st.plotly_chart(fig_count, use_container_width=True)

                with col_right:
                    st.subheader("Phân bổ giá trị khách hàng")
                    view_option = st.selectbox("🔍 Chọn góc nhìn phân tích:", ["Tần suất vs Chi tiêu", "Độ mới vs Chi tiêu"])
                    x_val = "Frequency" if "Tần suất" in view_option else "Recency"
                    
                    fig_scat = px.scatter(df_seg, x=x_val, y="Monetary", color="KMeans_Cluster", 
                                          color_discrete_map=cluster_colors,
                                          log_y=True, template="plotly_white", opacity=0.7)
                    st.plotly_chart(fig_scat, use_container_width=True)

                # --- 3. CHI TIẾT CÁC NHÓM (CẬP NHẬT TÊN NHÓM 0, 1...) ---
                st.markdown("---")
                st.subheader("📊 Đặc điểm chi tiết & Chiến lược")
                
                cluster_info = {
                    "0": {"name": "Nhóm 0: 👑 Nhóm VIP / Trung thành", "color": "#E59866", "desc": "Mua hàng rất thường xuyên, chi tiêu cực cao và vừa mới mua hàng gần đây.", "strategy": "Tặng quà tri ân, mời tham gia CLB khách hàng thân thiết, cung cấp dịch vụ ưu tiên."},
                    "1": {"name": "Nhóm 1: ⭐ Nhóm Tiềm năng", "color": "#7FB3D5", "desc": "Chi tiêu ở mức khá nhưng chưa mua hàng thường xuyên hoặc đã bắt đầu có dấu hiệu giãn cách thời gian mua.", "strategy": "Gửi voucher giảm giá giới hạn thời gian, giới thiệu sản phẩm mới dựa trên sở thích."},
                    "2": {"name": "Nhóm 2: 💤 Nhóm Ngủ đông", "color": "#D5D8DC", "desc": "Đã từng mua nhiều nhưng rất lâu rồi không quay lại. Có rủi ro rời bỏ hệ thống cao.", "strategy": "Chiến dịch Win-back: 'Chúng tôi nhớ bạn', tặng mã giảm giá sâu để kích cầu quay lại."},
                    "3": {"name": "Nhóm 3: 🌱 Nhóm Mới / Vãng lai", "color": "#A9DFBF", "desc": "Khách hàng mới phát sinh giao dịch hoặc mua rất ít, giá trị đơn hàng thấp.", "strategy": "Gửi hướng dẫn sử dụng, tặng mã giảm giá cho đơn hàng thứ 2 để tạo thói quen."}
                }

                for cluster, info in cluster_info.items():
                    with st.expander(f"{info['name']}"):
                        # Thêm vạch màu bên trái để phân biệt các nhóm đẹp mắt hơn
                        st.markdown(f"""
                        <div style="line-height: 1.6; font-size: 16px; border-left: 6px solid {info['color']}; padding-left: 15px; background-color: #fcfcfc; padding-top: 10px; padding-bottom: 10px; border-radius: 0 5px 5px 0;">
                            <p><strong>📝 Đặc điểm:</strong> {info['desc']}</p>
                            <p style="color: #2E86C1; margin-bottom: 0;"><strong>🚀 Chiến lược:</strong> {info['strategy']}</p>
                        </div>
                        """, unsafe_allow_html=True)

    # ==========================================
    # TAB 2: DỰ ĐOÁN NHÓM KHÁCH HÀNG MỚI
    # ==========================================
    with tab_predict:
        st.subheader("🎯 Dự đoán phân cụm cho dữ liệu mới")
        
        if kmeans_pipeline is None:
            st.error("⚠️ Không tìm thấy file `models/kmeans_pipeline.joblib`.")
        else:
            col_upload, col_req = st.columns([1.2, 1])
            with col_upload:
                uploaded_new_data = st.file_uploader("Chọn file CSV khách hàng mới", type=["csv"], key="upload_predict")
            
            with col_req:
                st.markdown("""
                <div style="background-color: #EAF4FE; padding: 15px; border-radius: 10px; border: 1px solid #B6D4FE;">
                    <h5 style="margin:0; color: #0D6EFD;">📋 Yêu cầu cột dữ liệu:</h5>
                    <p style="font-size: 14px; margin-bottom:5px; margin-top: 8px; color: #052C65;">File cần chứa đúng 3 cột sau:</p>
                    <ul style="font-size: 14px; color: #052C65;">
                        <li><b style="color: #0D6EFD;">Recency</b>: Số ngày từ lần mua cuối</li>
                        <li><b style="color: #0D6EFD;">Frequency</b>: Tổng số đơn hàng</li>
                        <li><b style="color: #0D6EFD;">Monetary</b>: Tổng số tiền</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            if uploaded_new_data is not None:
                new_df = pd.read_csv(uploaded_new_data)
                required_cols = ['Recency', 'Frequency', 'Monetary']
                
                if all(col in new_df.columns for col in required_cols):
                    with st.spinner("AI đang phân cụm..."):
                        # Dự đoán
                        predictions = kmeans_pipeline.predict(new_df[required_cols])
                        # Ép kiểu sang string để ăn màu chuẩn
                        new_df['KMeans_Cluster'] = predictions.astype(str) 
                        
                        st.success("✅ Phân nhóm hoàn tất!")
                        
                        # --- HIỂN THỊ BIỂU ĐỒ TRÒN MINH HỌA ---
                        col_chart, col_table = st.columns([1.2, 1])
                        
                        with col_chart:
                            st.write("📊 **Tỉ lệ phân bổ các nhóm dự đoán:**")
                            pred_counts = new_df['KMeans_Cluster'].value_counts().reset_index()
                            pred_counts.columns = ['Nhóm', 'Số lượng']
                            
                            # Áp dụng màu từ dictionary
                            fig_pie = px.pie(pred_counts, values='Số lượng', names='Nhóm', 
                                             color='Nhóm', color_discrete_map=cluster_colors,
                                             hole=0.4)
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col_table:
                            st.write("📋 **Bảng thống kê nhanh:**")
                            st.dataframe(pred_counts, hide_index=True, use_container_width=True)

                        # Hiển thị dữ liệu chi tiết
                        st.markdown("**🔍 Xem trước 10 dòng kết quả đầu tiên:**")
                        st.dataframe(new_df.head(10), use_container_width=True)
                        
                        # Nút tải xuống
                        csv = new_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Tải file kết quả (.csv)", data=csv, 
                                         file_name="data/predicted_segments.csv", mime="text/csv")
                else:
                    st.error("⚠️ File thiếu các cột cần thiết (Recency, Frequency, Monetary).")

elif selected == "Khuyến nghị":
    st.header("🎁 Hệ thống Khuyến nghị Cá nhân hóa")
    st.markdown("Dự đoán nhu cầu mua sắm dựa trên thuật toán **Matrix Factorization**.")

    # Gọi tài nguyên đã load
    model_svd, p_df = load_recommender_assets()

    if model_svd is not None and p_df is not None:
        
        # --- Form Tìm kiếm (Nhấn Enter để chạy) ---
        user_id = st.text_input("Nhập Customer Unique ID (Nhấn Enter để tìm):", placeholder="Ví dụ: 8d50f5eadf50201ccdcedfb9e2ac8455")
        
        # --- Nút Gợi ý ID ngẫu nhiên ---
        if st.button("🎲 Gợi ý 5 ID Khách hàng ngẫu nhiên"):
            import random
            try:
                n_users = model_svd.trainset.n_users
                random_inner_ids = random.sample(range(n_users), 5)
                random_raw_ids = [model_svd.trainset.to_raw_uid(i) for i in random_inner_ids]
                
                st.info("💡 **Hãy copy một trong các ID dưới đây và dán vào ô tìm kiếm:**\n" + 
                        "\n".join([f"* `{uid}`" for uid in random_raw_ids]))
            except Exception as e:
                st.warning("⚠️ Không thể trích xuất ID từ mô hình lúc này.")
        
        st.markdown("---")

        # --- Xử lý dự đoán khi ô user_id có dữ liệu ---
        if user_id:
            with st.spinner('🎯 Đang truy xuất dữ liệu và tính toán...'):
                
                # ==========================================
                # PHẦN 1: LỊCH SỬ MUA HÀNG (GIỮ NGUYÊN GIAO DIỆN BIG CARD & SAO SHOPEE)
                # ==========================================
                try:
                    inner_uid = model_svd.trainset.to_inner_uid(user_id)
                    user_history = model_svd.trainset.ur[inner_uid]

                    if user_history:
                        st.markdown("### 🛒 Sản phẩm khách hàng đã từng mua")
                        hist_data = []
                        for inner_iid, rating in user_history:
                            raw_iid = model_svd.trainset.to_raw_iid(inner_iid)
                            try:
                                p_name = p_df.loc[p_df['product_id'] == raw_iid, 'product_category_name'].values[0]
                            except (IndexError, KeyError):
                                p_name = "Sản phẩm không rõ tên"
                            hist_data.append((p_name, rating))

                        hist_data.sort(key=lambda x: x[1], reverse=True)

                        for name, rating in hist_data[:5]:
                            gold_stars = int(round(rating))
                            if gold_stars > 5: gold_stars = 5
                            if gold_stars < 1: gold_stars = 1
                            gray_stars = 5 - gold_stars
                            
                            # Mã HTML tạo sao có màu
                            stars_html = f'<span style="color: #ffce3d; font-size: 22px;">{"★" * gold_stars}</span>'
                            stars_html += f'<span style="color: #d1d5db; font-size: 22px;">{"★" * gray_stars}</span>'
                            
                            html_metric = f"""
                            <div style="display: flex; margin-bottom: 16px; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.06); min-height: 90px;">
                                <div style="flex: 7; background-color: #f8fafc; padding: 15px 25px; color: #1e293b; font-weight: 600; font-size: 18px; border-right: 3px solid #ffffff; display: flex; align-items: center;">
                                    🛍️ {name}
                                </div>
                                <div style="flex: 3; background-color: #fffbeb; padding: 15px 20px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                                    <div style="margin-bottom: 4px; letter-spacing: 2px;">{stars_html}</div>
                                    <div style="color: #b45309; font-weight: bold; font-size: 16px;">{rating:.1f} / 5.0</div>
                                </div>
                            </div>
                            """
                            st.markdown(html_metric, unsafe_allow_html=True)
                        
                        if len(hist_data) > 5:
                            st.caption(f"*... và {len(hist_data) - 5} sản phẩm khác trong lịch sử.*")
                        
                        st.write("") 
                except ValueError:
                    st.info("👋 Khách hàng này có vẻ là người mới, chưa có lịch sử mua hàng trong hệ thống.")

                # ==========================================
                # PHẦN 2: DỰ ĐOÁN ĐỀ XUẤT (BẢNG TRẢ VỀ DẠNG SỐ ĐƠN GIẢN)
                # ==========================================
                
                all_products = p_df['product_id'].tolist()
                predictions = [(p_id, model_svd.predict(user_id, p_id).est) for p_id in all_products]
                top_10 = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]

                res_df = pd.DataFrame(top_10, columns=['product_id', 'Điểm dự báo'])
                
                # Hàm đơn giản chỉ để format số
                def format_score(score):
                    return f"{score:.1f} / 5.0"

                try:
                    final_df = pd.merge(res_df, p_df[['product_id', 'product_category_name']], on='product_id', how='left')
                    final_df = final_df.rename(columns={
                        'product_id': 'Mã Sản phẩm',
                        'product_category_name': 'Tên Sản phẩm'
                    })
                    
                    # Áp dụng format điểm
                    final_df['Đánh giá dự kiến'] = final_df['Điểm dự báo'].apply(format_score)
                    final_df = final_df[['Mã Sản phẩm', 'Tên Sản phẩm', 'Đánh giá dự kiến']]
                
                except KeyError:
                    st.warning("Không tìm thấy cột 'product_category_name' trong file data/product_list.")
                    res_df['Đánh giá dự kiến'] = res_df['Điểm dự báo'].apply(format_score)
                    final_df = res_df[['product_id', 'Đánh giá dự kiến']].rename(columns={'product_id': 'Mã Sản phẩm'})

                # Hiển thị kết quả bằng bảng mặc định của Streamlit (Sạch sẽ, gọn gàng)
                st.success(f"Top 10 sản phẩm phù hợp nhất cho khách hàng này:")
                st.dataframe(final_df, use_container_width=True)
                
                # Hiệu ứng toast
                st.toast("✅ Đã hoàn tất tính toán gợi ý!", icon="🎉")
                
    else:
        st.error("⚠️ Không thể tải mô hình. Vui lòng kiểm tra xem file 'models/svd_model.pkl' và 'data/product_list.csv' đã được copy vào cùng thư mục với app.py chưa.")

    with st.expander("📝 Giải thích cơ chế"):
        st.write("Hệ thống phân tích các yếu tố ẩn từ lịch sử đánh giá để đưa ra gợi ý cá nhân hóa dựa trên từng khách hàng.")

elif selected == "Xu hướng":
    st.header("📈 Phân tích Xu hướng Mua sắm (FP-Growth)")
    
    # --- CSS CUSTOMIZATION ---
    # Ép kiểu cho các con số metric to hơn, in đậm và đổi màu
    st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        font-size: 45px !important;
        font-weight: 900 !important;
        color: #005f6b !important; /* Màu xanh tone-sur-tone với tiêu đề sidebar */
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("Khám phá các luật kết hợp: **Khách hàng mua sản phẩm A thường có xu hướng mua kèm sản phẩm B**.")

    # Expander giải thích ý nghĩa chỉ số
    with st.expander("📖 Hướng dẫn đọc chỉ số"):
        st.markdown("""
        - **Độ tin cậy (Confidence):** Xác suất khách mua B khi đã mua A. Ví dụ: 10% nghĩa là cứ 100 người mua A thì có 10 người mua thêm B.
        - **Độ nâng (Lift):** Mức độ liên quan giữa A và B. **Lift > 1** là quy luật có ý nghĩa thực tế (A thực sự thúc đẩy B). Lift càng cao, sự kết hợp càng mạnh.
        """)

    try:
        # --- LOAD DATA BẰNG JOBLIB ---
        import joblib
        # Đổi tên file thành file joblib bạn đang có
        rules_df = joblib.load('models/association_rules.joblib')
        
        # CHUẨN HÓA DỮ LIỆU TỪ JOBLIB (Xử lý tên cột và frozenset)
        # 1. Đổi tên cột từ tiếng Anh (nếu có) sang tiếng Việt để khớp với UI của bạn
        rename_dict = {
            'antecedents': 'Sản phẩm mua (A)',
            'consequents': 'Sản phẩm mua kèm (B)',
            'support': 'Độ phổ biến (Support)',
            'confidence': 'Độ tin cậy (Confidence)',
            'lift': 'Độ nâng (Lift)'
        }
        rules_df = rules_df.rename(columns=rename_dict)
        
        # 2. Phá vỡ định dạng frozenset của FP-Growth để hiển thị thành chuỗi (string)
        if 'Sản phẩm mua (A)' in rules_df.columns:
            rules_df['Sản phẩm mua (A)'] = rules_df['Sản phẩm mua (A)'].apply(lambda x: ', '.join(list(x)) if isinstance(x, (frozenset, set, list)) else str(x))
        if 'Sản phẩm mua kèm (B)' in rules_df.columns:
            rules_df['Sản phẩm mua kèm (B)'] = rules_df['Sản phẩm mua kèm (B)'].apply(lambda x: ', '.join(list(x)) if isinstance(x, (frozenset, set, list)) else str(x))

        # --- KHU VỰC BỘ LỌC ---
        st.markdown("### ⚙️ Bộ lọc tương tác")
        
        col1, col2 = st.columns(2)
        
        if not rules_df.empty:
            max_lift_val = float(rules_df['Độ nâng (Lift)'].max()) + 0.5
            max_conf_val = float(rules_df['Độ tin cậy (Confidence)'].max())
        else:
            max_lift_val = 5.0  
            max_conf_val = 1.0
            st.warning("⚠️ File dữ liệu đang trống.")

        with col1:
            min_conf = st.slider("Độ tin cậy tối thiểu (Confidence):", 
                                 min_value=0.0, max_value=max_conf_val, value=0.01, step=0.01)
        with col2:
            min_lift = st.slider("Độ nâng tối thiểu (Lift):", 
                                 min_value=0.0, max_value=max_lift_val, value=1.0, step=0.1)

        st.markdown("<br>", unsafe_allow_html=True) # Thêm khoảng trắng nhỏ cho thoáng

        # --- XỬ LÝ LỌC DỮ LIỆU ---
        filtered_rules = rules_df[
            (rules_df['Độ tin cậy (Confidence)'] >= min_conf) & 
            (rules_df['Độ nâng (Lift)'] >= min_lift)
        ].sort_values(by='Độ nâng (Lift)', ascending=False)

        # --- KHU VỰC HIỂN THỊ KẾT QUẢ BẮT MẮT ---
        if not filtered_rules.empty:
            # 1. Thể hiện KPI tổng quan
            c1, c2, c3 = st.columns(3)
            c1.metric(label="🔍 Tổng số quy luật", value=f"{len(filtered_rules)}")
            c2.metric(label="🚀 Lift cao nhất", value=f"{filtered_rules['Độ nâng (Lift)'].max():.2f}")
            c3.metric(label="🎯 Confidence cao nhất", value=f"{filtered_rules['Độ tin cậy (Confidence)'].max():.2%}")
            
            st.markdown("---")

            # 2. Chia Tab
            tab1, tab2 = st.tabs(["🔥 Top Combo Gợi Ý", "🛒 Tra Cứu Theo Danh Mục"])

            with tab1:
                # Highlight luật tốt nhất
                top_rule = filtered_rules.iloc[0]
                st.success(f"**✨ COMBO TIỀM NĂNG NHẤT:** Khách mua **{top_rule['Sản phẩm mua (A)']}** có khả năng rất cao sẽ mua kèm **{top_rule['Sản phẩm mua kèm (B)']}** (Độ nâng Lift: {top_rule['Độ nâng (Lift)']:.2f})")
                
                st.write("📋 **Bảng dữ liệu chi tiết (Màu càng đậm, tỷ lệ càng cao)**")
                
                styled_df = filtered_rules.style.background_gradient(
                    cmap='Greens', subset=['Độ nâng (Lift)']
                ).background_gradient(
                    cmap='Blues', subset=['Độ tin cậy (Confidence)']
                ).format({
                    'Độ phổ biến (Support)': '{:.5f}',
                    'Độ tin cậy (Confidence)': '{:.2%}', 
                    'Độ nâng (Lift)': '{:.3f}'
                })
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

            with tab2:
                st.markdown("##### 🔍 Tìm kiếm sản phẩm mua kèm theo ý muốn")
                list_categories = sorted(rules_df['Sản phẩm mua (A)'].unique().tolist())
                
                col_sel1, col_sel2 = st.columns([1, 1])
                with col_sel1:
                    selected_cat = st.selectbox("Chọn sản phẩm khách đang xem:", ["Tất cả"] + list_categories)
                
                if selected_cat != "Tất cả":
                    specific_rules = rules_df[rules_df['Sản phẩm mua (A)'] == selected_cat].sort_values(by='Độ nâng (Lift)', ascending=False)
                    if not specific_rules.empty:
                        st.info(f"💡 **Gợi ý Upsell/Cross-sell:** Khi khách hàng bỏ **{selected_cat}** vào giỏ, hãy đề xuất ngay các món sau:")
                        
                        st.table(specific_rules[['Sản phẩm mua kèm (B)', 'Độ tin cậy (Confidence)', 'Độ nâng (Lift)']].head(5).style.format({
                            'Độ tin cậy (Confidence)': '{:.2%}', 
                            'Độ nâng (Lift)': '{:.2f}'
                        }))
                    else:
                        st.warning("📉 Chưa tìm thấy dữ liệu mua kèm đủ mạnh cho danh mục này.")

        else:
            st.warning("Với mức lọc hiện tại, không có quy luật nào thỏa mãn. Hãy thử kéo thanh trượt giảm chỉ số xuống nhé!")

    except FileNotFoundError:
        st.error("⚠️ Không tìm thấy file 'models/association_rules.joblib'. Vui lòng kiểm tra lại tên file và đảm bảo nó nằm cùng thư mục với app.")
        
elif selected == "Dự đoán":
    # Tiêu đề to, đồng bộ với các trang khác
    st.title("🔮 Trợ lý AI - Dự báo Mức độ Hài lòng")
    st.markdown("---")
    st.write("Nhập thông tin đơn hàng bên dưới để AI dự báo khách hàng sẽ đánh giá bao nhiêu sao.")

    MODEL_PATH = "models/rf_model.joblib"
    
    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ Hệ thống AI chưa được huấn luyện. Vui lòng vào trang **Admin** để huấn luyện mô hình trước!")
    else:
        @st.cache_resource
        def load_ai_model():
            return joblib.load(MODEL_PATH)
        
        model = load_ai_model()

        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.subheader("📝 Thông số đơn hàng")
            with st.form("predict_form"):
                p = st.number_input("Giá sản phẩm (BRL):", min_value=0.0, value=150.0, step=10.0)
                f = st.number_input("Phí vận chuyển (BRL):", min_value=0.0, value=20.0, step=5.0)
                c = st.number_input("Số lượng sản phẩm:", min_value=1, value=1, step=1)
                d = st.slider("Số ngày giao hàng dự kiến:", min_value=1, max_value=60, value=5)
                
                submit = st.form_submit_button("🚀 Bắt đầu dự báo", use_container_width=True)

        with col2:
            st.subheader("📊 Kết quả dự báo")
            if submit:
                input_data = np.array([[p, f, c, d]])
                
                pred = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0]
                confidence = np.max(prob)

                if pred == 1:
                    st.success("🎯 **DỰ BÁO: HÀI LÒNG (4-5 ⭐)**")
                    st.write("Khả năng cao khách hàng sẽ để lại đánh giá tích cực.")
                    st.balloons()
                else:
                    st.error("⚠️ **DỰ BÁO: KHÔNG HÀI LÒNG (1-3 ⭐)**")
                    st.write("Cảnh báo rủi ro! Hãy xem xét giảm phí ship hoặc giao hàng nhanh hơn.")
                
                st.metric(label="Độ tự tin của AI (Confidence)", value=f"{confidence*100:.1f}%")
                st.progress(float(confidence))

elif selected == "Admin":
    st.title("⚙️ Admin Panel - Quản trị Mô hình AI")
    st.markdown("---")

    # 1. KHỞI TẠO TRẠNG THÁI ĐĂNG NHẬP (Nếu chưa có)
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    # 2. MÀN HÌNH ĐĂNG NHẬP (Nếu chưa đăng nhập)
    if not st.session_state['logged_in']:
        st.info("🔒 Vui lòng đăng nhập để truy cập khu vực Quản trị. (Gợi ý mật khẩu: 123456)")
        
        # Tạo form đăng nhập nhỏ gọn ở giữa trang
        col_space1, col_login, col_space2 = st.columns([1, 2, 1])
        with col_login:
            with st.form("login_form"):
                password = st.text_input("🔑 Nhập mật khẩu:", type="password")
                submit_login = st.form_submit_button("Đăng nhập", use_container_width=True)
                
                if submit_login:
                    if password == "123456":
                        st.session_state['logged_in'] = True
                        st.success("✅ Đăng nhập thành công! Đang tải hệ thống...")
                        st.rerun() # Lệnh này giúp tải lại trang ngay lập tức để vào màn hình Admin
                    else:
                        st.error("❌ Mật khẩu không đúng. Vui lòng thử lại!")
                        
    # 3. MÀN HÌNH ADMIN CHÍNH (Chỉ hiện khi đã đăng nhập)
    if st.session_state['logged_in']:
        
        # Nút Đăng xuất đặt ở góc phải
        col_title, col_logout = st.columns([4, 1])
        with col_title:
            st.success("👋 Chào mừng Quản trị viên!")
        with col_logout:
            if st.button("🚪 Đăng xuất", use_container_width=True):
                st.session_state['logged_in'] = False
                st.rerun()

        st.info("Khu vực dành cho Quản trị viên: Cập nhật dữ liệu mới và Retrain (Huấn luyện lại) AI.")

        tab1, tab2 = st.tabs(["📂 Cập nhật Dữ liệu & Retrain", "📈 Báo cáo Đánh giá AI"])

        with tab1:
            st.subheader("1. Upload dữ liệu mới")
            uploaded_file = st.file_uploader("Tải lên file dữ liệu (.csv) có chứa cột 'target'", type=['csv'])
            
            if uploaded_file is not None:
                df_new = pd.read_csv(uploaded_file)
                st.success("✅ Đã tải dữ liệu thành công!")
                st.dataframe(df_new.head())

                st.subheader("2. Tiến hành Huấn luyện (Retrain)")
                if st.button("🧠 Bắt đầu Huấn luyện AI", type="primary"):
                    with st.spinner('Đang xử lý dữ liệu và huấn luyện Random Forest...'):
                        try:
                            cols = ['price', 'freight_value', 'item_count', 'delivery_days', 'target']
                            df_train = df_new.dropna(subset=cols)
                            
                            X = df_train[['price', 'freight_value', 'item_count', 'delivery_days']]
                            y = df_train['target'].astype(int)

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

                            model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
                            model.fit(X_train, y_train)

                            joblib.dump(model, "models/rf_model.joblib")
                            
                            y_pred = model.predict(X_test)
                            acc = accuracy_score(y_test, y_pred)
                            
                            st.cache_resource.clear()

                            st.success(f"🎉 Huấn luyện thành công! Độ chính xác trên tập kiểm tra: **{acc*100:.2f}%**")
                            st.info("👉 Hãy chuyển sang tab 'Báo cáo Đánh giá AI' để xem chi tiết biểu đồ.")
                            
                            st.session_state['y_test'] = y_test
                            st.session_state['y_pred'] = y_pred
                            st.session_state['feature_importances'] = model.feature_importances_
                            st.session_state['feature_names'] = X.columns
                            
                        except Exception as e:
                            st.error(f"❌ Có lỗi xảy ra trong quá trình huấn luyện: {e}")

        with tab2:
            st.subheader("Báo cáo Hiệu suất Mô hình (Model Performance)")
            if 'y_test' not in st.session_state:
                st.warning("Chưa có dữ liệu báo cáo. Hãy Upload file và Retrain mô hình ở Tab bên cạnh trước.")
            else:
                y_test = st.session_state['y_test']
                y_pred = st.session_state['y_pred']
                importances = st.session_state['feature_importances']
                features = st.session_state['feature_names']

                col_metric, col_cm = st.columns([1, 2])
                
                with col_metric:
                    acc = accuracy_score(y_test, y_pred)
                    st.metric("Độ chính xác tổng thể (Accuracy)", f"{acc*100:.2f}%")
                    st.markdown("**Báo cáo phân loại chi tiết:**")
                    report = classification_report(y_test, y_pred)
                    st.text(report)

                with col_cm:
                    st.write("**Ma trận nhầm lẫn (Confusion Matrix)**")
                    cm = confusion_matrix(y_test, y_pred)
                    x_labels = ['Dự đoán: Kém (0)', 'Dự đoán: Tốt (1)']
                    y_labels = ['Thực tế: Kém (0)', 'Thực tế: Tốt (1)']
                    fig_cm = ff.create_annotated_heatmap(z=cm, x=x_labels, y=y_labels, colorscale='Blues', showscale=True)
                    fig_cm.update_layout(margin=dict(t=30, l=100))
                    st.plotly_chart(fig_cm, use_container_width=True)

                st.markdown("---")
                st.subheader("🌟 Mức độ quan trọng của các yếu tố (Feature Importance)")
                df_imp = pd.DataFrame({'Yếu tố': features, 'Mức độ quan trọng': importances})
                df_imp = df_imp.sort_values(by='Mức độ quan trọng', ascending=True)
                
                fig_imp = px.bar(df_imp, x='Mức độ quan trọng', y='Yếu tố', orientation='h', 
                                 color='Mức độ quan trọng', color_continuous_scale='Viridis')
                st.plotly_chart(fig_imp, use_container_width=True)
