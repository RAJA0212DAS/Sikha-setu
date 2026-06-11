from flask import Flask, request, jsonify, send_from_directory
from rag_engine import RAGEngine
from document_loader import load_documents
from config import DATA_PATH

app = Flask(__name__)

# Initialize RAG
rag = RAGEngine()

# =========================
# FRONTEND ROUTES
# =========================

# Home page
@app.route("/")
def home():
    return send_from_directory("frontend", "index.html")

# Serve ALL frontend files (html, css, js)
@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory('frontend', path)


# =========================
# BACKEND ROUTES
# =========================

# Load documents
@app.route("/load", methods=["POST"])
def load_docs():
    docs = load_documents(DATA_PATH)
    print("DOCS LOADED:", len(docs))

    if not docs:
        return jsonify({"error": "No documents found!"})

    rag.build_index(docs)
    return jsonify({"status": "Documents loaded successfully"})


# Ask question
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        query = data.get("question")

        if not query:
            return jsonify({"error": "No question provided"})

        answer = rag.query(query)

        return jsonify({"answer": answer})

    except Exception as e:
        print("ERROR IN /ask:", str(e))
        return jsonify({"error": str(e)})


# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":
    app.run(debug=True)