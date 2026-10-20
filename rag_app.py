from httpcore import __name
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env
load_dotenv()

# Load the google genai api key
api_key = os.getenv("GEMINI-API-KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing from .env")

# Loading the pdf contents
print("\nLoading the pdf.....\n")
loader = PyPDFLoader("TechCorp_Official_Employee_Handbook.pdf")
document = loader.load()
# print(document[0].page_content)

# Chunking the text
print("\nChunking text.....\n")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(document)
# print(chunks[0].page_content)
# print(len(chunks))

# Creating vector database
print("\nCreating the vector databases...\n")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Configure the database to act as a document retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

template = """
Use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know.
Use three senetnces maximum and keeps the answer concise.

context: {context}

question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template)
# print(prompt)

# Initilize the free gemini model tier
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    google_api_key=api_key
)

# Helper function to build a single string chunks separated by 2 nextlines
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Connecting everything together using Langchain Expresssion Language (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Tesing
"""
user_question = "What days can I work from home?"
print(f"Question: {user_question}")

response = rag_chain.invoke(user_question)
if isinstance(response.content, list):
    clean_text = response.content[0]['text']
else:
    clean_text = response.content

print(f"Answer: {clean_text}")
"""


if __name__ == "__main__":
    
    # Continuous chatting
    print("\n-------------Chatbot started------------")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        # 1. User inputs
        user_input = input("\nYour question: ")

        # 2. Checking whether the input is break statement
        if user_input in ['exit', 'quit']:
            break

        # 3. Send the user input to the RAG pipeline
        response = rag_chain.invoke(user_input)

        # 4. Retrieving the text part only from the response if the response is a list
        if isinstance(response.content, list):
            clean_text = response.content[0]['text']
        else:
            clean_text = response.content

        # 5. Printing the LLM response
        print(f"Answer: {clean_text}")