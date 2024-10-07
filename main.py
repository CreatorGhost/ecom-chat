import os
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from uuid import uuid4
from pinecone import Pinecone, ServerlessSpec
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Load environment variables
load_dotenv()

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

index_name = "ecommerce-index"  # Define your index name

# Create Pinecone index if it doesn't exist
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,  # Adjust dimension based on your embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Adjust region as needed
    )

# Load the CSV file
loader = CSVLoader('ecommerce_dataset.csv')
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,  # Adjusted to handle most responses in one chunk
    chunk_overlap=150,  # Provides overlap to maintain context
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.split_documents(data)

# Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# Initialize Pinecone vector store
# vector_store = PineconeVectorStore.from_documents(
#     texts,
#     index_name=index_name,
#     embedding=embeddings,
# )

print("Loading vector store")
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5},
)

print("Loading retriever")

# Define a function to format documents
print("Loading format docs")
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])




template = """Answer the question based only on the following context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
)
answer = chain.invoke(" i have to add fucking items to the cart where could i do it")
print(answer.content)