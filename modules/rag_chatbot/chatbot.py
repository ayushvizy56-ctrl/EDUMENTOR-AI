from modules.rag_chatbot.pdf_ingestion import load_pdf, load_text
from modules.rag_chatbot.vector_store import build_vector_store, load_vector_store
from modules.rag_chatbot.rag_chain import ask
from config.settings import EMBED_DIR


class EduMentorChatbot:
    """
    Stateful chatbot that remembers conversation history
    and answers questions from the student's own study material.

    Usage:
        bot = EduMentorChatbot()
        bot.load_document("notes.pdf")
        response = bot.chat("What is Newton's second law?")
        print(response["answer"])
    """

    def __init__(self):
        self.store        = None
        self.history      = []
        self.store_name   = None

    def load_document(self, pdf_path: str, store_name: str = None) -> str:
        """
        Load a PDF and build its vector store.
        If already indexed, loads from disk (fast).
        """
        name = store_name or pdf_path.replace("/", "_").replace("\\", "_").replace(".pdf", "")
        index_path = EMBED_DIR / f"{name}.index"

        if index_path.exists():
            print(f"Loading existing index for '{name}'...")
            self.store = load_vector_store(name)
        else:
            print(f"Building new index for '{name}'...")
            chunks     = load_pdf(pdf_path)
            self.store = build_vector_store(chunks, name)

        self.store_name = name
        self.history    = []
        return f"Ready! Loaded {self.store['index'].ntotal} chunks from your document."

    def load_text_notes(self, text: str, name: str = "my_notes") -> str:
        """Load raw text notes instead of a PDF."""
        chunks     = load_text(text, source_name=name)
        self.store = build_vector_store(chunks, name)
        self.store_name = name
        self.history    = []
        return f"Ready! Indexed {len(chunks)} chunks from your notes."

    def chat(self, question: str) -> dict:
        """Ask a question and get an answer grounded in your study material."""
        if self.store is None:
            return {
                "answer":  "Please load a document first using load_document().",
                "sources": [],
            }

        result = ask(question, self.store, self.history)

        # Save to history for multi-turn conversation
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant",  "content": result["answer"]})

        return result

    def clear_history(self):
        """Reset conversation but keep the document loaded."""
        self.history = []
        print("Conversation cleared.")


# Quick test — run this file directly to test the chatbot
if __name__ == "__main__":
    bot = EduMentorChatbot()

    # Test with raw text (no PDF needed)
    bot.load_text_notes("""
    Newton's Laws of Motion:

    First Law: An object at rest stays at rest, and an object in motion 
    stays in motion unless acted upon by an external force. This is also 
    called the law of inertia.

    Second Law: Force equals mass times acceleration (F = ma). 
    The acceleration of an object depends on the net force acting on it 
    and its mass.

    Third Law: For every action there is an equal and opposite reaction.
    When you push a wall, the wall pushes back with equal force.
    """, name="newton_laws")

    print("\nChatbot ready. Testing...\n")

    questions = [
        "What is Newton's second law?",
        "Can you give me an example of the third law?",
        "What does inertia mean?",
    ]

    for q in questions:
        print(f"Student: {q}")
        result = bot.chat(q)
        print(f"EduMentor: {result['answer']}")
        print(f"Sources: {result['sources']}\n")