import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.rag_chatbot.chatbot import EduMentorChatbot

st.set_page_config(page_title="Study Chatbot", page_icon="💬", layout="wide")
st.title("💬 AI Study Chatbot")
st.markdown("Ask questions about your study material. The AI answers only from your notes.")

# Initialize chatbot once per session
if "chatbot" not in st.session_state:
    st.session_state.chatbot  = EduMentorChatbot()
    st.session_state.messages = []
    st.session_state.doc_loaded = False

# Sidebar — load material
with st.sidebar:
    st.subheader("📚 Load Study Material")
    input_type = st.radio("Input type", ["Type/Paste Notes", "Upload PDF"])

    if input_type == "Type/Paste Notes":
        notes = st.text_area("Paste your notes here:", height=200,
                             placeholder="Enter your study notes...")
        doc_name = st.text_input("Topic name:", value="my_notes")
        if st.button("Load Notes") and notes:
            with st.spinner("Indexing your notes..."):
                msg = st.session_state.chatbot.load_text_notes(notes, name=doc_name)
                st.session_state.doc_loaded = True
                st.session_state.messages   = []
            st.success(msg)

    else:
        uploaded = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded and st.button("Load PDF"):
            path = f"data/raw/{uploaded.name}"
            os.makedirs("data/raw", exist_ok=True)
            with open(path, "wb") as f:
                f.write(uploaded.read())
            with st.spinner("Reading and indexing PDF..."):
                msg = st.session_state.chatbot.load_document(path)
                st.session_state.doc_loaded = True
                st.session_state.messages   = []
            st.success(msg)

    if st.session_state.doc_loaded:
        if st.button("Clear Chat"):
            st.session_state.chatbot.clear_history()
            st.session_state.messages = []

# Chat interface
if not st.session_state.doc_loaded:
    st.info("👈 Load your study material from the sidebar to start chatting.")
else:
    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("📖 Sources"):
                    for s in msg["sources"]:
                        st.caption(f"Page {s['page']} of {s['source']} (relevance: {s['score']:.0%})")

    # Input
    if prompt := st.chat_input("Ask a question about your notes..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chatbot.chat(prompt)
            st.write(result["answer"])
            if result.get("sources"):
                with st.expander("📖 Sources"):
                    for s in result["sources"]:
                        st.caption(f"Page {s['page']} of {s['source']} (relevance: {s['score']:.0%})")

        st.session_state.messages.append({
            "role":    "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
        })