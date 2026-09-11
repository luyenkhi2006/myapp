import json
import requests
import faiss
import streamlit as st

from sentence_transformers import SentenceTransformer


# =========================
# CONFIG
# =========================

VECTOR_DB = "vectorstore/index.faiss"
CHUNKS_FILE = "vectorstore/chunks.json"

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "llama3.2"


# =========================
# LOAD DATA
# =========================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_vector_database():

    index = faiss.read_index(
        VECTOR_DB
    )

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return index, data["chunks"], data["metadata"]


model = load_embedding_model()

index, chunks, metadata = load_vector_database()


# =========================
# PAGE
# =========================

st.set_page_config(
    page_title="RAG Document QA",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖 Hệ thống hỏi đáp tài liệu sử dụng RAG")

st.write(
    "Hệ thống tìm kiếm Top-K chunks trước khi "
    "gọi LLM để sinh câu trả lời."
)


# =========================
# SIDEBAR
# =========================

st.sidebar.header("⚙️ Cấu hình")

top_k = st.sidebar.slider(
    "Số lượng Top-K chunks",
    min_value=1,
    max_value=10,
    value=3
)


# =========================
# THỐNG KÊ
# =========================

st.header("📊 Thông tin Vector Database")

col1, col2 = st.columns(2)

# Lấy danh sách file
files = set(
    item["file_name"]
    for item in metadata
)

col1.metric(
    "Số file",
    len(files)
)

col2.metric(
    "Số chunks",
    len(chunks)
)


# =========================
# CHUNK MẪU
# =========================

st.header("📄 Chunk mẫu")

sample_chunk = 0

if len(chunks) > 0:

    st.code(
        chunks[sample_chunk],
        language="text"
    )

    st.write(
        metadata[sample_chunk]
    )


# =========================
# METADATA
# =========================

st.header("🏷️ Metadata")

if len(metadata) > 0:

    st.json(
        metadata[0]
    )


# =========================
# QUESTION
# =========================

st.header("❓ Đặt câu hỏi")

question = st.text_input(
    "Nhập câu hỏi:",
    placeholder="Ví dụ: Điều kiện nào được bảo hành?"
)


search_button = st.button(
    "🔍 Tìm kiếm Top-K"
)


# =========================
# SEARCH
# =========================

if search_button:

    if not question.strip():

        st.warning(
            "Vui lòng nhập câu hỏi."
        )

    else:

        # Embedding câu hỏi
        query_embedding = model.encode(
            [question],
            convert_to_numpy=True
        )


        # Search FAISS
        distances, indices = index.search(
            query_embedding,
            top_k
        )


        # Lưu kết quả vào session
        results = []

        for rank, idx in enumerate(indices[0]):

            result = {
                "rank": rank + 1,
                "chunk_id": int(idx),
                "score": float(distances[0][rank]),
                "text": chunks[idx],
                "metadata": metadata[idx]
            }

            results.append(result)


        st.session_state["results"] = results

        st.session_state["question"] = question


# =========================
# DISPLAY TOP-K
# =========================

if "results" in st.session_state:

    st.header("🔎 Top-K Chunks")

    results = st.session_state["results"]

    for result in results:

        st.subheader(
            f"🥇 Rank {result['rank']} - "
            f"Chunk {result['chunk_id']}"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"**File:** "
                f"{result['metadata']['file_name']}"
            )

            st.write(
                f"**Page:** "
                f"{result['metadata']['page']}"
            )

        with col2:

            st.write(
                f"**Score:** "
                f"{result['score']:.4f}"
            )

            st.write(
                f"**Chunk index:** "
                f"{result['metadata']['chunk_index']}"
            )


        st.info(
            result["text"]
        )

        st.divider()


# =========================
# CALL LLM
# =========================

if "results" in st.session_state:

    st.header("🤖 Sinh câu trả lời")

    if st.button(
        "🚀 Gọi LLM và trả lời"
    ):

        question = st.session_state["question"]

        results = st.session_state["results"]


        # Tạo context
        context = ""

        for result in results:

            context += f"""
Nguồn:
File: {result['metadata']['file_name']}
Trang: {result['metadata']['page']}

Nội dung:
{result['text']}

"""


        prompt = f"""
Bạn là trợ lý hỏi đáp tài liệu.

Chỉ sử dụng thông tin trong CONTEXT
để trả lời câu hỏi.

Nếu context không có thông tin phù hợp,
hãy nói rằng không tìm thấy thông tin
trong tài liệu.

Không tự bịa thông tin.

CÂU HỎI:
{question}

CONTEXT:
{context}

Hãy trả lời ngắn gọn, chính xác.
"""


        # Gọi Ollama
        try:

            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )


            if response.status_code == 200:

                answer = response.json()["response"]

                st.subheader(
                    "💬 Câu trả lời"
                )

                st.success(answer)


                # =========================
                # SOURCES
                # =========================

                st.subheader(
                    "📚 Nguồn tham khảo"
                )

                for i, result in enumerate(results):

                    file_name = result[
                        "metadata"
                    ]["file_name"]

                    page = result[
                        "metadata"
                    ]["page"]

                    st.write(
                        f"[{i+1}] "
                        f"{file_name} - "
                        f"Trang {page}"
                    )


            else:

                st.error(
                    "Không thể gọi LLM."
                )


        except Exception as e:

            st.error(
                f"Lỗi kết nối Ollama: {e}"
            )