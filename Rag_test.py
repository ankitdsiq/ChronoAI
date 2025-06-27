from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters.base import TextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

load_dotenv()

base_url="http://172.16.10.2:11434"

loader = TextLoader("text.txt",encoding="utf-8")
docs = loader.load()
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


splitter = RecursiveCharacterTextSplitter(
chunk_size=1000,
chunk_overlap=50
)

chunks = splitter.create_documents([docs[0].page_content])


vector_store = Chroma.from_documents(
documents=chunks,
embedding=embedding,
collection_name="my_chroma",
persist_directory="./chroma_db",  
)

class RAG:
    def __init__(self):
        pass
        
    def invoke(self,query: str):
        print(query)

        template = PromptTemplate(
            template="""
            You are a helpful assistant. Answer only from the provided content.

            Context:
            {context}

            Question:
            {question}

        Instructions:
        - Include citations from the context  where the information was found.
        - If the answer is not present in the context, simply respond with: "Not in content".
            """,
            input_variables=["context","question"]
        )

        chat_model = ChatOllama(model='llama3.3:latest',base_url=base_url)
        
        retriver = vector_store.as_retriever(search_type="similarity",search_kwargs={'k':10})
        
        retrived_doc  = retriver.invoke(query)

        context_text = " ".join(doc.page_content for doc in retrived_doc)

        final_prompt = template.invoke({'context':context_text,"question":query})
        res = dict()

        ans = chat_model.invoke(final_prompt)
        res['output'] = ans.content
        return res
