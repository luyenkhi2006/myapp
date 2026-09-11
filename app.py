import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import os

# Cấu hình API Key (Nên dùng st.secrets trong thực tế)
os.environ["OPENAI_API_KEY"] = "NHAP_API_KEY_CUA_BAN_VAO_DAY"

st.title("Hệ thống RAG Tra cứu Chính sách & Hướng dẫn")

# 1. Chuẩn bị & Tải lên 3-5 file PDF
uploaded_files = st.file_uploader("Tải lên các file PDF (Bảo hành, Đổi trả, HDSD)", type="pdf", accept_multiple_files=True)

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
                
    # 2. Xử lý Chunking & Hiển thị thông số
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.create_documents(text_data, metadatas=metadata_list)
    
    st.success("Đã xử lý tài liệu thành công!")
    col1, col2 = st.columns(2)
    col1.metric("Số lượng file PDF", len(uploaded_files))
    col2.metric("Tổng số Chunks", len(chunks))
    
    with st.expander("Xem Chunk mẫu và Metadata"):
        if chunks:
            st.write("**Nội dung Chunk đầu tiên:**")
            st.info(chunks[0].page_content)
            st.write("**Metadata:**", chunks[0].metadata)

    # Khởi tạo Vector Store (Chroma hoặc FAISS)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    st.divider()
    
    # 3. Nhập câu hỏi và hiển thị Top-K
    query = st.text_input("Nhập câu hỏi của bạn về chính sách hoặc sản phẩm:")
    k = st.slider("Số lượng chunks (top-k) muốn trích xuất:", min_value=1, max_value=5, value=3)
    
    if query:
        docs = vector_store.similarity_search(query, k=k)
        
        st.subheader(f"Top {k} Chunks tìm thấy (Trích xuất trước khi gọi LLM)")
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i+1}** (Nguồn: {doc.metadata['source']}, Trang: {doc.metadata['page']})")
            st.caption(doc.page_content)
            
        # 4. Gọi LLM và hiển thị câu trả lời có nguồn
        st.subheader("Câu trả lời từ LLM")
        with st.spinner("Đang tổng hợp câu trả lời..."):
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=docs, question=query)
            
            st.write(answer)
            
            # Trích xuất nguồn duy nhất để hiển thị
            sources = set([doc.metadata['source'] for doc in docs])
            st.markdown(f"**Nguồn tham khảo:** {', '.join(sources)}")
