import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="EMIRIX | UAE University Library",
    page_icon="📚",
    layout="wide"
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }
body, .stApp { background-color: #f5f5f5; color: #1a1a1a; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

.sidebar {
    width: 280px; min-width: 280px;
    background: linear-gradient(180deg, #8b0000 0%, #5c0000 100%);
    display: flex; flex-direction: column; padding: 0;
    box-shadow: 4px 0 15px rgba(0,0,0,0.2);
}
.sidebar-logo {
    padding: 30px 24px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.15);
    display: flex; flex-direction: column; align-items: flex-start;
}
.sidebar-logo img {
    width: 120px;
    margin-bottom: 12px;
    filter: brightness(0) invert(1);
}
.sidebar-title {
    font-size: 22px; font-weight: 700;
    color: white; letter-spacing: 2px; margin: 0;
}
.sidebar-sub {
    font-size: 11px; color: rgba(255,255,255,0.7);
    letter-spacing: 1px; margin-top: 4px; text-transform: uppercase;
}
.sidebar-nav { padding: 20px 16px; flex: 1; }
.nav-label {
    font-size: 10px; color: rgba(255,255,255,0.5);
    text-transform: uppercase; letter-spacing: 1.5px;
    padding: 0 8px; margin-bottom: 8px;
}
.nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 12px; border-radius: 8px;
    color: rgba(255,255,255,0.8);
    font-size: 14px; cursor: pointer; margin-bottom: 4px;
}
.nav-item:hover, .nav-item.active {
    background: rgba(255,255,255,0.15); color: white;
}
.sidebar-footer {
    padding: 16px 24px;
    border-top: 1px solid rgba(255,255,255,0.15);
    font-size: 11px; color: rgba(255,255,255,0.5); text-align: center;
}
.uaeu-badge {
    background: rgba(255,255,255,0.1); border-radius: 6px;
    padding: 8px 12px; margin: 16px; font-size: 11px;
    color: rgba(255,255,255,0.8); text-align: center;
    border: 1px solid rgba(255,255,255,0.15);
}
.chat-topbar {
    background: white; padding: 16px 28px;
    border-bottom: 1px solid #e5e5e5;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.topbar-title { font-size: 16px; font-weight: 600; color: #1a1a1a; }
.topbar-status { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #666; }
.status-dot {
    width: 8px; height: 8px; background: #22c55e;
    border-radius: 50%; animation: pulse 2s infinite;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

.welcome-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; padding: 40px;
}
.welcome-icon-big { font-size: 56px; margin-bottom: 16px; }
.welcome-h { font-size: 22px; font-weight: 700; color: #1a1a1a; margin-bottom: 8px; }
.welcome-p { font-size: 14px; color: #666; line-height: 1.7; max-width: 420px; margin-bottom: 28px; }
.suggestion-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 10px; max-width: 480px; margin: 0 auto;
}
.suggestion-card {
    background: white; border: 1px solid #e5e5e5;
    border-radius: 10px; padding: 12px 14px;
    font-size: 13px; color: #333; text-align: left;
    border-left: 3px solid #8b0000;
}
.suggestion-card .s-icon { font-size: 18px; margin-bottom: 6px; }
.suggestion-card .s-text { font-size: 12px; color: #444; }

.msg-wrap-user { display: flex; justify-content: flex-end; margin-bottom: 16px; }
.msg-wrap-bot {
    display: flex; justify-content: flex-start;
    margin-bottom: 4px; align-items: flex-start; gap: 10px;
}
.bot-avatar {
    width: 34px; height: 34px; background: #8b0000;
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 700; color: white; flex-shrink: 0; margin-top: 2px;
}
.msg-user {
    background: #8b0000; color: white;
    padding: 12px 16px; border-radius: 18px 18px 4px 18px;
    max-width: 65%; font-size: 14px; line-height: 1.6;
    box-shadow: 0 2px 8px rgba(139,0,0,0.2);
}
.msg-bot {
    background: white; color: #1a1a1a;
    padding: 14px 18px; border-radius: 18px 18px 18px 4px;
    max-width: 70%; font-size: 14px; line-height: 1.7;
    border: 1px solid #e5e5e5; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.voice-btn {
    background: none; border: 1px solid #e5e5e5;
    border-radius: 20px; padding: 4px 10px;
    font-size: 12px; color: #888; cursor: pointer;
    margin-left: 44px; margin-bottom: 8px;
    display: inline-flex; align-items: center; gap: 4px;
    transition: all 0.2s;
}
.voice-btn:hover { background: #fff0f0; border-color: #8b0000; color: #8b0000; }

.citation-box {
    background: #fff8f8; border-left: 3px solid #8b0000;
    padding: 8px 12px; border-radius: 0 6px 6px 0;
    font-size: 12px; color: #666; margin-top: 4px;
    max-width: 70%; margin-left: 44px; margin-bottom: 12px;
}
.citation-label {
    font-size: 10px; color: #8b0000;
    text-transform: uppercase; letter-spacing: 1px;
    font-weight: 600; margin-bottom: 4px;
}
.citation-item { margin: 2px 0; padding-left: 6px; }

.stTextInput > div > div > input {
    background: #f9f9f9 !important; color: #1a1a1a !important;
    border-radius: 12px !important; border: 1.5px solid #e5e5e5 !important;
    padding: 12px 18px !important; font-size: 14px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #8b0000 !important;
    box-shadow: 0 0 0 3px rgba(139,0,0,0.08) !important;
    background: white !important;
}
.stTextInput > div > div > input::placeholder { color: #aaa !important; }
.stButton > button {
    background: #8b0000 !important; color: white !important;
    border-radius: 12px !important; border: none !important;
    padding: 12px 28px !important; font-size: 14px !important;
    font-weight: 600 !important; width: 100% !important;
}
.stButton > button:hover {
    background: #a50000 !important;
    box-shadow: 0 4px 12px rgba(139,0,0,0.25) !important;
}
.input-hint { font-size: 11px; color: #aaa; text-align: center; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)


# ── TTS JavaScript ────────────────────────────────────────────
def tts_button(text, idx):
    safe_text = text.replace("'", "\\'").replace('"', '\\"').replace('\n', ' ')
    return (
        f'<button class="voice-btn" onclick="'
        f'var u=new SpeechSynthesisUtterance(\'{safe_text}\');'
        f'u.lang=\'ar-AE\';'
        f'window.speechSynthesis.cancel();'
        f'window.speechSynthesis.speak(u);">🔊 Listen</button>'
    )


# ── Load RAG ─────────────────────────────────────────────────
@st.cache_resource
def load_rag():
    persist_directory = r"C:\Users\RTX\OneDrive\Desktop\EMIRIX RAG\vector_DB"
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20}
    )
    os.environ["GOOGLE_API_KEY"] = "AIzaSyB_i3N1xzj7NrLNl79yWyeq3B4bybj8FQg"
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    system_prompt = (
        "You are Emirix, a professional Library Assistant "
        "for the UAE University Library Repository. "
        "Answer in the same language the user uses (Arabic or English). "
        "Use ONLY the retrieved context to answer. "
        "If not found say: I don't have this in the UAE University Library. "
        "Be concise, academic, and helpful.\n\nContext: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, chain)


# ── APA Citation ─────────────────────────────────────────────
def build_apa_citation(metadata):
    author    = metadata.get("author", "")
    year      = metadata.get("year", "")
    title     = metadata.get("title", "")
    publisher = metadata.get("publisher", "")
    page      = metadata.get("page", None)
    if not author or not title:
        for key, value in metadata.items():
            if not author and any(x in key.lower() for x in ["author", "writer"]):
                author = value
            if not title and any(x in key.lower() for x in ["title", "name"]):
                title = value
    author = author or "Unknown Author"
    year   = year   or "n.d."
    title  = title  or "Untitled"
    citation = f"{author}. ({year}). {title}."
    if publisher:
        citation += f" {publisher}."
    if page:
        citation += f" p. {int(page) + 1}"
    return citation


# ── Session State ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_key" not in st.session_state:
    st.session_state.input_key = 0


# ── Layout ────────────────────────────────────────────────────
col_side, col_main = st.columns([1, 3])

with col_side:
    # ── ضع مسار الـ logo هنا ──────────────────────────────
    # logo_path = "logo.png"   ← ضع ملف الـ logo في نفس مجلد EmirixApp.py
    # logo_b64  = base64.b64encode(open(logo_path,"rb").read()).decode()
    # logo_tag  = f'<img src="data:image/png;base64,{logo_b64}">'
    # أو لو عندك URL:
    # logo_tag  = '<img src="https://your-logo-url.png">'

    import base64
    from PIL import Image
    import io
    _ico = Image.open(r"C:\Users\RTX\Downloads\IMG-20260502-WA0007.ico").convert("RGBA")
    _buf = io.BytesIO()
    _ico.save(_buf, format="PNG")
    ico_b64 = base64.b64encode(_buf.getvalue()).decode()
    logo_tag = (
        "<div style=\"width:64px;height:64px;border-radius:14px;overflow:hidden;"
        "box-shadow:0 4px 16px rgba(0,0,0,0.4);margin-bottom:14px;"
        "border:2px solid rgba(255,255,255,0.2);\">"
        f'<img src="data:image/png;base64,{ico_b64}" style="width:100%;height:100%;object-fit:cover;">' 
        "</div>"
    )

    st.markdown(f"""
    <div class="sidebar">
        <div class="sidebar-logo">
            {logo_tag}
            <p class="sidebar-title">EMIRIX</p>
            <p class="sidebar-sub">Library Intelligence</p>
        </div>
        <div class="uaeu-badge">🏛️ UAE University<br>Library Repository</div>
        <div class="sidebar-nav">
            <p class="nav-label">Navigation</p>
            <div class="nav-item active">💬 &nbsp; Chat Assistant</div>
            <div class="nav-item">📚 &nbsp; Browse Collection</div>
            <div class="nav-item">🔍 &nbsp; Advanced Search</div>
            <div class="nav-item">📄 &nbsp; Research Papers</div>
            <div class="nav-item">👤 &nbsp; My Account</div>
        </div>
        <div class="sidebar-footer">
            Powered by Gemini & LangChain<br>
            © 2025 EMIRIX · UAEU
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_main:
    st.markdown("""
    <div class="chat-topbar">
        <div>
            <div class="topbar-title">📚 Library Assistant</div>
            <div style="font-size:12px;color:#888;margin-top:2px">
                Ask about books, research papers, and academic resources
            </div>
        </div>
        <div class="topbar-status">
            <div class="status-dot"></div>
            Online · Ready
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-state">
            <div class="welcome-icon-big">🏛️</div>
            <div class="welcome-h">Welcome to EMIRIX</div>
            <div class="welcome-p">
                Your intelligent gateway to the UAE University Library Repository.<br>
                Search books, research papers, and academic resources in Arabic and English.
            </div>
            <div class="suggestion-grid">
                <div class="suggestion-card">
                    <div class="s-icon">📖</div>
                    <div class="s-text">Find a book by title or author</div>
                </div>
                <div class="suggestion-card">
                    <div class="s-icon">🔬</div>
                    <div class="s-text">Search research papers by topic</div>
                </div>
                <div class="suggestion-card">
                    <div class="s-icon">🌐</div>
                    <div class="s-text">Search in Arabic or English</div>
                </div>
                <div class="suggestion-card">
                    <div class="s-icon">📑</div>
                    <div class="s-text">Get APA citations instantly</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-wrap-user">'
                f'<div class="msg-user">{msg["content"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="msg-wrap-bot">'
                f'<div class="bot-avatar">E</div>'
                f'<div class="msg-bot">{msg["content"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            # ── Voice Button ──────────────────────────────
            st.markdown(tts_button(msg["content"], i), unsafe_allow_html=True)

            if msg.get("citations"):
                items = "".join(
                    f'<div class="citation-item">📄 {c}</div>'
                    for c in msg["citations"]
                )
                st.markdown(
                    f'<div class="citation-box">'
                    f'<div class="citation-label">📚 Sources</div>'
                    f'{items}'
                    f'</div>',
                    unsafe_allow_html=True
                )

    user_input = st.text_input(
        "",
        placeholder="Ask about a book, author, or research topic...",
        label_visibility="collapsed",
        key=f"input_{st.session_state.input_key}"
    )

    if st.button("Send  ➔"):
        if user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("Searching the library..."):
                rag_chain = load_rag()
                result    = rag_chain.invoke({"input": user_input})
                answer    = result["answer"]
                citations = [
                    build_apa_citation(doc.metadata)
                    for doc in result.get("context", [])
                ]
            st.session_state.messages.append({
                "role":      "assistant",
                "content":   answer,
                "citations": citations
            })
            st.session_state.input_key += 1
            st.rerun()

    st.markdown(
        "<p class='input-hint'>Press Send · Supports Arabic & English · 🔊 Voice available</p>",
        unsafe_allow_html=True
    )