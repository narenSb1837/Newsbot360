
import os
os.environ["COHERE_API_KEY"]='1WDphHnJYzXRcm2EjDcvyqXRnKRG6n83XxX7LPFx'
os.environ['PINECONE_API_KEY']='f840e6fa-f34e-412d-8da1-b20eff50d688'#'6dbebefb-e722-4241-8041-00f56ca935ca'
os.environ['PINECONE_ENV']='gcp-starter'
os.environ['QDRANT_API_KEY']='B2p7WN_t2TIpugdRgeZ-S5ApOPZ-VigWZZxhxDE036aBbATU_mpx1g'
os.environ['GOOGLE_API_KEY']='AIzaSyAUggwhrE0LoTBDWrfeU6kxQuxA0FP6eCk'
os.environ['APIFY_API_TOKEN']='apify_api_K90vlEcLcKMx43KED0DpKQuxz2cTUr2CXPtv'
os.environ['HUGGINGFACEHUB_API_TOKEN'] ='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'
os.environ['VOYAGE_API_KEY']='pa-yEmOi9CYAehyiFGbGJKRUwVxUfkdlNdXoqIulWYzNKs'
os.environ['HUGGINGFACEHUB_API_TOKEN'] ='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'

import cohere
import langchain
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
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.cache import SQLiteCache
import voyageai
from langchain_community.embeddings import VoyageEmbeddings


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

        
def vector_embedding(file_name=None):
    if file_name:
        print('2..')
        with open(file_name, encoding='utf-8') as f:
            state_of_the_union = f.read()
            text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            separators=['\n \n'],
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
            )
            docs = text_splitter.create_documents([state_of_the_union])
    #model_name = "WhereIsAI/UAE-Large-V1"
    llm_chain=create_hypothetical_chain()
    #hf = HuggingFaceEmbeddings(model_name=model_name)

    embeddings = VoyageEmbeddings(voyage_api_key='pa-yEmOi9CYAehyiFGbGJKRUwVxUfkdlNdXoqIulWYzNKs')
    #base_embeddings = CohereEmbeddings(model="multilingual-22-12")
    '''embeddings = HypotheticalDocumentEmbedder(
    llm_chain=llm_chain,
    base_embeddings=base_embeddings)'''
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),environment=os.getenv("PINECONE_ENV"))
    index_name = "trial"
    #docsearch = Pinecone.from_existing_index(index_name, base_embeddings)
    # This is a long document we can split up.
    if file_name:
        print('embeddings started')

        docsearch = Pinecone.from_documents(docs,embeddings, index_name=index_name)
        print('embeddings end')
    else:
          
        docsearch = Pinecone.from_existing_index(index_name, embeddings)
    create_cache()
    return docsearch

