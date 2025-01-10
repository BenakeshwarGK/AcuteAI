from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from googletrans import Translator
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Flask app initialization
app = Flask(__name__)

# Environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Set API keys as environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Function to download embeddings using Sentence Transformers
def download_hugging_face_embeddings():
    print("Downloading embeddings model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model

# Initialize embeddings model
embeddings_model = download_hugging_face_embeddings()

# Pinecone index name
index_name = "medicalbot"

# Initialize text-generation pipeline
generator = pipeline("text-generation", model="gpt2", device=-1)  # Use CPU

# Initialize Pinecone VectorStore
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings_model
)

# Set up retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=generator,
    chain_type="map_reduce",
    retriever=retriever
)

# Function to translate text to Kannada
def translate_to_kannada(text):
    translator = Translator()
    translated = translator.translate(text, src='en', dest='kn')
    return translated.text

# Flask routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print(f"User input: {msg}")

    # Generate response
    response = qa_chain.run(msg)
    print(f"Generated response: {response}")

    # Translate response to Kannada
    translated_response = translate_to_kannada(response)
    print(f"Translated response (Kannada): {translated_response}")

    return translated_response

if __name__ == '__main__':
    print("Starting the Flask application...")
    app.run(host="0.0.0.0", port=8080, debug=True)
