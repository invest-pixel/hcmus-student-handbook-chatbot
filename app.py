import os
from dotenv import load_dotenv

import streamlit as st

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

DB_PATH = "chroma_db"


@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    return vector_db


@st.cache_resource
def load_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1,
        request_timeout=30
    )

    return llm


def ask_question(question, vector_db, llm, k=3):
    retriever = vector_db.as_retriever(
        search_kwargs={"k": k}
    )

    docs = retriever.invoke(question)

    context_parts = []

    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "không rõ")
        if isinstance(page, int):
            page += 1

        context_parts.append(
            f"[Nguồn {i} - Trang {page}]\n{doc.page_content[:1200]}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
Bạn là chatbot hỗ trợ sinh viên HCMUS.

Quy tắc:
- Chỉ trả lời dựa trên TÀI LIỆU được cung cấp.
- Không tự bịa quy định.
- Nếu tài liệu không có thông tin, hãy nói: "Không tìm thấy thông tin này trong sổ tay sinh viên."
- Trả lời bằng tiếng Việt.
- Trả lời rõ ràng, ngắn gọn, dễ hiểu.
- Cuối câu trả lời ghi nguồn trang nếu có.

TÀI LIỆU:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:
"""

    response = llm.invoke(prompt)

    return response.content, docs


st.set_page_config(
    page_title="Chatbot HCMUS",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Chatbot Sổ tay Sinh viên HCMUS")
st.caption("Hỏi đáp dựa trên sổ tay sinh viên. Nó không thay phòng đào tạo, nhưng ít nhất không bắt mày xếp hàng.")

if not os.path.exists(DB_PATH):
    st.error("Chưa tìm thấy thư mục chroma_db. Hãy tạo vector database trong notebook trước.")
    st.stop()

if not os.getenv("GOOGLE_API_KEY"):
    st.error("Không tìm thấy GOOGLE_API_KEY trong file .env.")
    st.stop()

vector_db = load_vector_db()
llm = load_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Nhập câu hỏi về sổ tay sinh viên...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra sổ tay sinh viên..."):
            answer, docs = ask_question(question, vector_db, llm)

        st.markdown(answer)

        with st.expander("📚 Nguồn được truy xuất"):
            for i, doc in enumerate(docs, start=1):
                page = doc.metadata.get("page", "không rõ")
                if isinstance(page, int):
                    page += 1

                st.markdown(f"**Đoạn {i} - Trang {page}**")
                st.write(doc.page_content[:1500])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })