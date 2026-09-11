import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

st.set_page_config(page_title="RAG Tra cứu Tài liệu", layout="wide")
st.title("📄 Hệ thống RAG Tra cứu Chính sách & Hướng dẫn")

# Ô nhập API Key bên sidebar
with st.sidebar:
    st.header("⚙️ Cấu hình")
    api_key = st.text_input("Nhập Google Gemini API Key:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

# 1. Tải lên các file PDF
uploaded_files = st.file_uploader(
    "Tải lên các file PDF (Chính sách bảo hành, Đổi trả, HDSD)",
    type="pdf",
    accept_multiple_files=True,
)

if uploaded_files:
    text_data = []
    metadata_list = []

    for file in uploaded_files:
        reader = PdfReader(file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                text_data.append(text)
                metadata_list.append({"source": file.name, "page": i + 1})

    # 2. Xử lý Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.create_documents(text_data, metadatas=metadata_list)

    st.success(f"Đã xử lý xong {len(uploaded_files)} tài liệu!")
    col1, col2 = st.columns(2)
    col1.metric("Số lượng file PDF", len(uploaded_files))
    col2.metric("Tổng số Chunks", len(chunks))

    with st.expander("🔍 Xem Chunk mẫu và Metadata"):
        if chunks:
            st.write("**Nội dung Chunk đầu tiên:**")
            st.info(chunks[0].page_content)
            st.write("**Metadata:**", chunks[0].metadata)

    if not api_key:
        st.warning(
            "⚠️ Vui lòng nhập Google Gemini API Key ở thanh bên (Sidebar) để bắt đầu tìm kiếm."
        )
    else:
        try:
            # Khởi tạo Vector Store với Gemini Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", google_api_key=api_key
            )
            vector_store = FAISS.from_documents(chunks, embeddings)

            st.divider()

            # 3. Nhập câu hỏi và hiển thị Top-K
            query = st.text_input(
                "💬 Nhập câu hỏi của bạn về chính sách hoặc sản phẩm:"
            )
            k = st.slider(
                "Chọn số lượng chunks (top-k) muốn trích xuất:",
                min_value=1,
                max_value=5,
                value=3,
            )

            if query:
                docs = vector_store.similarity_search(query, k=k)

                st.subheader(
                    f"📌 Top {k} Chunks tìm thấy (Trích xuất trước khi gọi LLM)"
                )
                for i, doc in enumerate(docs):
                    st.markdown(
                        f"**Chunk {i+1}** *(Nguồn: `{doc.metadata['source']}`, Trang: {doc.metadata['page']})*"
                    )
                    st.info(doc.page_content)

                # 4. Gọi Gemini LLM
                st.subheader("🤖 Câu trả lời từ Gemini LLM")
                with st.spinner("Đang tổng hợp câu trả lời..."):
                    context = "\n\n".join(
                        [
                            f"[Nguồn: {doc.metadata['source']}, Trang: {doc.metadata['page']}]\n{doc.page_content}"
                            for doc in docs
                        ]
                    )

                    prompt = f"""Dựa vào nội dung tài liệu bên dưới để trả lời câu hỏi. Trả lời chính xác, ngắn gọn dựa trên thông tin được cung cấp.

Tài liệu tham khảo:
{context}

Câu hỏi: {query}
Trả lời:"""

                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        google_api_key=api_key,
                        temperature=0,
                    )
                    response = llm.invoke(prompt)

                    st.success(response.content)

                    sources = list(
                        set([doc.metadata["source"] for doc in docs])
                    )
                    st.markdown(
                        f"**📚 Nguồn tham khảo:** {', '.join(sources)}"
                    )
        except Exception as e:
            st.error(f"Lỗi khi xử lý API Key: {e}")
