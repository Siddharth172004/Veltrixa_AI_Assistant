from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load = PyPDFLoader("Test_1.pdf")
data = load.load()
#print(len(data))

split_text = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

doc = split_text.split_documents(data)

embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

document = FAISS.from_documents(doc,embed)
document.save_local("Vector Store")