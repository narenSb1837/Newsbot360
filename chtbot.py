import os
import cohere
import langchain

from langchain.llms import Cohere
os.environ["COHERE_API_KEY"]='1WDphHnJYzXRcm2EjDcvyqXRnKRG6n83XxX7LPFx'
os.environ['PINECONE_API_KEY']='f840e6fa-f34e-412d-8da1-b20eff50d688'#'6dbebefb-e722-4241-8041-00f56ca935ca'
os.environ['PINECONE_ENV']='gcp-starter'
os.environ['QDRANT_API_KEY']='B2p7WN_t2TIpugdRgeZ-S5ApOPZ-VigWZZxhxDE036aBbATU_mpx1g'
os.environ['GOOGLE_API_KEY']='AIzaSyAUggwhrE0LoTBDWrfeU6kxQuxA0FP6eCk'
os.environ['APIFY_API_TOKEN']='apify_api_K90vlEcLcKMx43KED0DpKQuxz2cTUr2CXPtv'
os.environ['HUGGINGFACEHUB_API_TOKEN'] ='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'

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.llms import Cohere
from langchain.prompts import PromptTemplate

from langchain.chains import RetrievalQA

import os
from dbase import *
from vect_embed import *
vectDB=vector_embedding()
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

def chtreply(query):

    from langchain.chains import ConversationalRetrievalChain
    llm=Cohere()
    from langchain.retrievers.multi_query import MultiQueryRetriever

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectDB.as_retriever(), llm=llm#must set qdrant embeddings as base embedding and not hypothetical embeddings else it will ask fr paid version
    )

#llm = OpenAI(temperature=0)
    compressor = CohereRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    system_template = """use only the following news to answer the question.Strictly answer only from the news below and don't try to answer from your knowledge.
    {context}
    
    """

        # Create the chat prompt templates
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
        ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)
    qa = ConversationalRetrievalChain.from_llm(llm,compression_retriever,combine_docs_chain_kwargs={"prompt": qa_prompt})

    latest_cht=get_chthistory()
    resp=qa.run({'question':query,'chat_history':latest_cht})
    insert_data(query,resp)
    return resp
