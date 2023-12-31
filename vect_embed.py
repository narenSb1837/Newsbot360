
import os
os.environ["COHERE_API_KEY"]='P7b3enduxOgEYHdhY4X0xBu7j7Q125sAZ3MMdSWy'
os.environ['PINECONE_API_KEY']='ffc34cb8-5e0e-446e-9a92-ac18c157b720'#'8644c0cf-1aef-4bbc-a39d-721e0cee986c'#c85ea1ee-1d0f-4c78-9285-1dd05c5194aa'
os.environ['PINECONE_ENV']='gcp-starter'
import cohere
import langchain
from langchain.chat_models import cohere
from langchain.document_loaders import PyPDFLoader
from langchain.llms import Cohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
import textwrap as tr
import random
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.vectorstores import pinecone,Pinecone
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.cache import SQLiteCache
print('VECTEMBED FILE')
def create_hypothetical_chain():
    
    prompt_template = """
    Please write a passage to answer the question
    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
    llm=Cohere()
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return llm_chain
def create_cache():
    langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

def vector_embedding():
    
    model_name = "WhereIsAI/UAE-Large-V1"
    llm_chain=create_hypothetical_chain()
    hf = HuggingFaceEmbeddings(model_name=model_name)
    embeddings = HypotheticalDocumentEmbedder(
    llm_chain=llm_chain,
    base_embeddings=hf)
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),environment=os.getenv("PINECONE_ENV"))
    index_name = "trial"
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    #pine_cone = Pinecone.from_documents(docs,embeddings, index_name=index_name)
    create_cache()
    return docsearch

