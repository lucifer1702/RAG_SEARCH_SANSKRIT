#RAG

import os
from dotenv import load_dotenv
load_dotenv()

# open ai api key
os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_API_KEY']

#imports
from llama_index.core import (
  VectorStoreIndex,
  SimpleDirectoryReader,
  StorageContext,
  load_index_from_storage
)
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
import os.path
import streamlit as sc 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    documents=SimpleDirectoryReader("data").load_data()
    index=VectorStoreIndex.from_documents(documents,show_progress=True)
    return index


def model(query):
    index=load_data()
    retriever = VectorIndexRetriever(index=index)
    post_processor=SimilarityPostprocessor(threshold=0.7)
    query_engine = RetrieverQueryEngine(retriever=retriever,node_postprocessors=[post_processor])
    response=query_engine.query(query)
    return response

    
    


def main():
    query = "What is the capital of France?"
    response = model(query)
    pprint_response(response)
    # print(response.get_top_answer())    
    
  
    
if __name__ == "__main__":
    main()