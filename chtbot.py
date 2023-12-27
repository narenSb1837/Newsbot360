import os
import cohere
import langchain
from langchain.chat_models import cohere

from langchain.llms import Cohere


from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.llms import Cohere
from langchain.prompts import PromptTemplate

from langchain.chains import RetrievalQA

import os
from dotenv import load_dotenv
load_dotenv()
from dbase import *
from vect_embed import *
vectDB=vector_embedding()
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

def chtreply(query):

    from langchain.chains import ConversationalRetrievalChain
    llm=Cohere()
    retriever=vectDB.as_retriever()

#llm = OpenAI(temperature=0)
    compressor = CohereRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    qa = ConversationalRetrievalChain.from_llm(llm,compression_retriever)

    latest_cht=get_chthistory()
    resp=qa.run({'question':query,'chat_history':latest_cht})
    insert_data(query,resp)
    return resp
