from groq import Groq
from config.settings import GROQ_API_KEY
from modules.rag_chatbot.vector_store import search

_client = Groq(api_key=GROQ_API_KEY)


def ask(question: str, store: dict, chat_history: list = None) -> dict:
    relevant_chunks = search(question, store, top_k=4)

    if not relevant_chunks:
        return {
            "answer":  "I couldn't find relevant information in your study material.",
            "sources": [],
            "context": [],
        }

    context_text = "\n\n---\n\n".join([
        f"[Page {c['page']} of {c['source']}]\n{c['text']}"
        for c in relevant_chunks
    ])

    history_text = ""
    if chat_history:
        for turn in chat_history[-4:]:
            role = "Student" if turn["role"] == "user" else "EduMentor"
            history_text += f"{role}: {turn['content']}\n"

    prompt = f"""You are EduMentor, an intelligent and friendly AI tutor.
Answer ONLY using the context below. If answer not in context, say "I don't see this in your notes".

STUDY MATERIAL:
{context_text}

RECENT CONVERSATION:
{history_text}

STUDENT QUESTION: {question}

YOUR ANSWER:"""

    response = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    answer = response.choices[0].message.content.strip()

    return {
        "answer":  answer,
        "sources": [
            {"page": c["page"], "source": c["source"], "score": c["relevance_score"]}
            for c in relevant_chunks
        ],
        "context": relevant_chunks,
    }