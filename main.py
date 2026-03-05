from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

parser = StrOutputParser()

memory = {}
def history(session_id: str):
    if session_id not in memory:
        memory[session_id] = InMemoryChatMessageHistory()
    return memory[session_id]

model = ChatOpenAI(
        model = "openrouter/free",
        openai_api_key = os.getenv("OPENROUTER_API_KEY"),
        openai_api_base = "https://openrouter.ai/api/v1",
        temperature= 0.7
    )
embedding = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")
data = FAISS.load_local("Vector Store",embedding, allow_dangerous_deserialization= True)
retriver = data.as_retriever(
    search_type= "mmr",
    search_kwargs= {"k" : 3} 
)

prompt = """You are Veltrixa Dynamics Internal Knowledge AI Assistant.

Your role is to assist users with information from the internal documentation of Veltrixa Dynamics Pvt. Ltd.

Follow these instructions strictly:

1. Greeting Handling
If the user sends a greeting (hello, hi, hey, good morning, good evening, etc.), respond with:
"Welcome to Veltrixa Dynamics Pvt. Ltd. Internal Knowledge Assistant. How can I assist you today?"

2. Context-Based Answers
For all company-related questions, answer ONLY using the provided context.

3. No External Knowledge
Do NOT use any external knowledge or assumptions.

4. Accuracy Requirement
Provide exact numbers, IDs, percentages, names, and terminology exactly as written in the context.

5. Missing Information
If the question is related to Veltrixa Dynamics but the answer is not present in the provided context, respond exactly with:
"Information not available in provided documents."

6. Out-of-Scope Queries
If the question is completely unrelated to Veltrixa Dynamics or its documentation, respond politely with:
"I can only assist with questions related to Veltrixa Dynamics internal documentation."

7. Tone
Maintain a professional enterprise assistant tone."""

f_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt),
        MessagesPlaceholder(variable_name= "history"),
        ("human", """
         Context : {context}
         Question : {Query}""")
         ])

#user = input("Ask Your Query : ")

def bot(user,session_id: str):
    
    docs = retriver.invoke(user)
    
    context = "\n\n".join([db.page_content for db in docs])
    
    chain = f_prompt | model | parser
    
    chat_memory = RunnableWithMessageHistory(
        chain,
        history,
        input_message_key= "Query",
        history_messages_key="history")

    result = chat_memory.invoke({"context" : context,"Query" : user},config={"configurable" : {"session_id" : session_id}})
    
    return result



