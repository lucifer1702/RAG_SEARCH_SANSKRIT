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


def load_data():
    PERSIST_DIR = "./storage"
    if not os.path.exists(PERSIST_DIR):
     docs=SimpleDirectoryReader("data").load_data()
     index=VectorStoreIndex.from_documents(docs)
     index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
     storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
     index = load_index_from_storage(storage_context)
     return index


def model(query):
    index=load_data()
    retriever = VectorIndexRetriever(index=index)
    post_processor=SimilarityPostprocessor(threshold=0.7)
    query_engine = RetrieverQueryEngine(retriever=retriever,node_postprocessors=[post_processor])
    response=query_engine.query(query)
    return response

    
    


def main():
    sc.title('RAG BASED SEARCH ENGINE')
    menu=['Home','About']
    choice=sc.sidebar.selectbox('Menu',menu)
    if choice=='Home':
        sc.subheader('query')
        with sc.form(key='RAG-SEARCH'):
            raw_query = sc.text_area("type-here")
            sub_text = sc.form_submit_button(label='YES')
        if sub_text:
          
            prediction=model(raw_query)
            sc.success("RAG SEARCH RESULT")
            sc.write(prediction)
    else:
         sc.subheader('About')
         sc.write('This is a search engine based on RAG model')
         sc.write('RAG model is a retrieval augmented generation model')
         sc.write('This project was done for the purpose of thesis evaluation and research')
         # response = model("What is the
              
            
            
            

    # print(response.get_top_answer())    
    
  
    
if __name__ == "__main__":
    main()