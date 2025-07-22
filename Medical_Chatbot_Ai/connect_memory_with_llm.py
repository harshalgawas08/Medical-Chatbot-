import os

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# If you're not using pipenv, uncomment these
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())

# Step 1: Setup LLM (Groq - LLaMA3 or Mixtral)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama3-8b-8192"  # or "mixtral-8x7b-32768"

def load_llm(model_name):
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=model_name,
        temperature=0.8
    )
    return llm

# Step 2: Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don’t know the answer, just say you don’t know. Don’t make up answers.
Only answer based on the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Step 3: Load FAISS DB + Embeddings
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Build RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(GROQ_MODEL_NAME),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=False,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Ask a question
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

# Step 6: Display Results
print("\nRESULT:\n", response["result"])
