import os
import torch
import transformers
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline

from streamlit_chat import message
import streamlit as st

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader


from PyPDF2 import PdfReader

from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document


# Charger le local_LLM
def load_LLM(name_model):
  tokenizer = AutoTokenizer.from_pretrained(name_model)
  model = AutoModelForCausalLM.from_pretrained(name_model,
                                                load_in_8bit=True, # charger en 8bit avec bitsandbytes
                                                device_map='auto', # utiliser les GPU configuration auto
                                                torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True) # AutoModelForSeq2SeqLM pour text2text-generation et AutoModelForCausalLM pour text-generation 

  pipe = pipeline("text-generation",
                  model=model,
                  tokenizer=tokenizer,
                  min_length=10,
                  max_length=1024,
                  temperature=0,
                  top_p=0.95,
                  repetition_penalty=1.15
                  )
  # on utilise HuggingFacePipeline 
  local_llm = HuggingFacePipeline(pipeline=pipe)
  return local_llm

# découper le document en petits morceaux
def load_document(raw_text):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, # controls the max size (in terms of number of characters) of the final documents
                                               chunk_overlap=200 # specifies how much overlap there should be between chunks.
                                              )
  texts = text_splitter.split_text(raw_text)# [:nombre_limite_de_morceaux]
  docs = [Document(page_content=t) for t in texts]
  return docs

# création de la chaine pour résumer le texte
def load_chain(chain_type):
    # prompt pour l'inférence pour chaque morceau de texte
    map_prompt = """
    Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    # prompt pour l'inférence finale (avec tous les outputs des map_prompt)
    combine_prompt = """
    Write a concise summary of the following text delimited by triple backquotes.
    Return your response in bullet points which covers the key points of the text.
    ```{text}```
    BULLET POINT SUMMARY:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    chain = load_summarize_chain(llm=st.session_state.local_llm,
                                chain_type=chain_type, 
                                map_prompt=map_prompt_template,
                                combine_prompt=combine_prompt_template,
                                verbose=True
                                )
    return chain

# extraction du contenu du document
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def main():
    st.set_page_config(page_title="PDF Summarizer 🤖")
    st.markdown("# Mon Générateur de Résumé Local ! 🤖📈")
    st.markdown("## Cette application - Summarizer utilise LangChain & Streamlit")

    # variables de session
    if "local_llm" not in st.session_state:
      st.session_state.local_llm = None

    if "chain" not in st.session_state:
      st.session_state.chain = None

    # demander à l'utilisateur un fichier
    pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    
    # découper le document en morceaux 
    raw_text =  get_pdf_text(pdf_docs)
    docs = load_document(raw_text)
    # afficher la liste des morceaux de texte
    st.write(docs)

    name_model = st.selectbox(
        'Choix du LLM HuggingFace',
        ('bigscience/bloomz-560m', 'bigscience/bloomz-1b1', 'bigscience/bloomz-3b'))

    chain_type = st.selectbox(
        'Quel type de chaine utiliser ?',
        ("map_reduce", "stuff", "refine"))

    # import du LLM
    if st.button('Importer'):
      with st.spinner("Loading..."):
        st.session_state.local_llm = load_LLM(name_model)

      st.write('Le modèle a été importé avec succès !')

      # creation de la chaine pour résumer
      st.session_state.chain = load_chain(chain_type)

    if st.button('Summerize !'):
      # affiche le contenu à résumer avec msg.page_content
      for msg in docs:
        message(message=msg.page_content, is_user=False)
      # affiche un message fictif
      message(message='Résume moi ce document.', is_user=True)
      with st.spinner("Loading..."):
        # utilise la chaine 
        resume = st.session_state.chain.run(docs)
      message(message=resume, is_user=False)

if __name__ == "__main__":
    main()
