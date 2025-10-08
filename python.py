import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    # Xử lý giá trị 0 cho mẫu số
    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini cho Nhận xét tổng quan (One-shot) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét tổng quan."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        # Nếu có lỗi 400, đây thường là lỗi cấu hình API hoặc giới hạn sử dụng
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Hàm gọi API Gemini cho Chatbot (Conversational) ---
def get_chat_response(prompt, context_data, api_key):
    """Gửi yêu cầu chat kèm theo lịch sử và ngữ cảnh dữ liệu."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'

        # 1. Tạo Context Prefix (System Instruction)
        context_prefix = (
            "Bạn là một chuyên gia phân tích tài chính. Hãy trả lời các câu hỏi của người dùng "
            "dựa trên dữ liệu tài chính đính kèm sau đây (chỉ dùng dữ liệu này để trả lời): \n\n"
            f"--- DỮ LIỆU TÀI CHÍNH CƠ SỞ ---\n{context_data}\n---------------------------\n"
            "Bây giờ, hãy trả lời câu hỏi sau của người dùng: "
        )
        
        contents = []
        
        # 2. Xử lý lịch sử chat: Lấy tất cả tin nhắn trừ tin nhắn user mới nhất (prompt)
        history_messages = st.session_state.messages[:-1] 

        for message in history_messages:
            role = None
            if message["role"] == "user":
                role = "user"
            elif message["role"] == "assistant":
                # Ánh xạ vai trò Streamlit 'assistant' sang vai trò Gemini 'model'
                role = "model" 
            
            # Chỉ thêm vào contents nếu vai trò hợp lệ và không phải là tin nhắn thông báo ban đầu
            if role and not message["content"].startswith("Vui lòng tải file Excel") and not message["content"].startswith("Dữ liệu đã được tải lên"):
                 contents.append({"role": role, "parts": [{"text": message["content"]}]})
        
        # 3. Thêm prompt mới nhất (user) VÀ GẮN NGỮ CẢNH vào nó
        # Việc này đảm bảo ngữ cảnh luôn được gửi và tránh lỗi role liên tiếp.
        final_prompt_with_context = context_prefix + prompt

        contents.append({"role": "user", "parts": [{"text": final_prompt_with_context}]})

        # Bắt đầu gọi API
        response = client.models.generate_content(
            model=model_name,
            contents=contents
        )
        return response.text

    except APIError as e:
        return f"Lỗi API: Vui lòng kiểm tra Khóa API hoặc vai trò (roles) trong lịch sử chat. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Lỗi không xác định khi chat: {e}"


# --- Khởi tạo State cho Chat và Ngữ cảnh Dữ liệu ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Vui lòng tải file Excel để bắt đầu phân tích và trò chuyện."}]

if "financial_data_context" not in st.session_state:
    st.session_state.financial_data_context = None

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    # Lấy API Key từ Secrets (Dùng chung cho cả one-shot và chat)
    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")
    
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            # Tính Chỉ số Thanh toán Hiện hành
            thanh_toan_hien_hanh_N = "N/A"
            thanh_toan_hien_hanh_N_1 = "N/A"
            delta = "N/A"

            try:
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán (Chỉ tính nếu mẫu số khác 0)
                if no_ngan_han_N != 0 and no_ngan_han_N_1 != 0:
                    thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                    thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                    delta = thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần" if isinstance(thanh_toan_hien_hanh_N_1, (int, float)) else "N/A"
                    )
                with col2:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần" if isinstance(thanh_toan_hien_hanh_N, (int, float)) else "N/A",
                        delta=f"{delta:.2f}" if isinstance(delta, (int, float)) else None
                    )
                    
            except IndexError:
                 st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
            except ZeroDivisionError:
                 st.warning("Không thể tính chỉ số Thanh toán Hiện hành do Nợ Ngắn Hạn bằng 0.")

            # --- Chuẩn bị dữ liệu cho AI (Ngữ cảnh chung cho cả one-shot và chat) ---
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{thanh_toan_hien_hanh_N_1}", 
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False) 

            # Cập nhật ngữ cảnh dữ liệu cho phiên chat
            st.session_state.financial_data_context = data_for_ai
            
            # --- Chức năng 5: Nhận xét AI Tổng quan (One-shot) ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI Tổng quan)")
            
            if st.button("Yêu cầu AI Phân tích"):
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                # else đã được xử lý ở trên

            # ------------------------------------------------------------------
            # --- CHỨC NĂNG MỚI: 6. KHUNG CHAT TƯƠNG TÁC ---
            # ------------------------------------------------------------------
            st.divider()
            st.subheader("6. Chatbot Tài chính Tương tác (Hỏi đáp chuyên sâu)")
            st.caption("Bạn có thể đặt các câu hỏi cụ thể về dữ liệu đã tải lên.")

            # Hiển thị lịch sử chat
            # Cần reset lịch sử khi người dùng tải file mới
            if len(st.session_state.messages) == 1 and st.session_state.messages[0]["content"].startswith("Vui lòng tải file"):
                st.session_state.messages = [{"role": "assistant", "content": "Dữ liệu đã được tải lên thành công! Hãy đặt câu hỏi đầu tiên của bạn."}]

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Xử lý input mới từ người dùng
            if prompt := st.chat_input("Hỏi Gemini về báo cáo này..."):
                # Thêm tin nhắn user vào lịch sử (trước khi gọi API)
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Hiển thị tin nhắn người dùng
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Gọi API và hiển thị phản hồi của AI
                if api_key:
                    with st.chat_message("assistant"):
                        with st.spinner("Đang xử lý yêu cầu..."):
                            # Gọi hàm chat với ngữ cảnh dữ liệu
                            full_response = get_chat_response(
                                prompt, 
                                st.session_state.financial_data_context, 
                                api_key
                            )
                            st.markdown(full_response)
                        
                        # Thêm phản hồi vào lịch sử
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error("Không có Khóa API để thực hiện chat.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
        st.session_state.messages = [{"role": "assistant", "content": f"Lỗi dữ liệu. Vui lòng tải file mới: {ve}"}]
        st.session_state.financial_data_context = None # Xóa ngữ cảnh lỗi
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")
        st.session_state.messages = [{"role": "assistant", "content": f"Lỗi xử lý file. Vui lòng tải file mới: {e}"}]
        st.session_state.financial_data_context = None

else:
    # Nếu chưa tải file, reset hoặc hiển thị thông báo ban đầu
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích và mở khóa khung chat.")
    if st.session_state.messages[0]["content"].startswith("Dữ liệu đã được tải lên"):
         st.session_state.messages = [{"role": "assistant", "content": "Vui lòng tải file Excel để bắt đầu phân tích và trò chuyện."}]
    
    st.divider()
    st.subheader("6. Chatbot Agribank Tài chính Tương tác (Hỏi đáp chuyên sâu)")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    st.chat_input("Hỏi Gemini về báo cáo này...", disabled=True)
