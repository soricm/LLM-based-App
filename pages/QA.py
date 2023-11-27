import os
import torch
import transformers
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from streamlit_chat import message
import streamlit as st

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

from langchain.embeddings import HuggingFaceInstructEmbeddings

from PyPDF2 import PdfReader

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

# découper le text (le document) en petits morceaux 
def load_document(docs):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, # controls the max size (in terms of number of characters) of the final documents
                                                chunk_overlap=200 # specifies how much overlap there should be between chunks.
                                                )
  texts = text_splitter.split_documents(docs)
  return texts

# création de la chaine
def qa(local_llm, vectordb):
  memory = ConversationBufferMemory(memory_key="chat_history", 
                                    input_key='question', 
                                    output_key='answer', 
                                    return_messages=True)
  
  qa = ConversationalRetrievalChain.from_llm(local_llm,
                                              retriever=vectordb.as_retriever(search_kwargs={"k": 3}), # top 3 des embeddings avec recherche par similarité
                                              memory=memory,
                                              return_source_documents=True,
                                              verbose=True
                                            )
  return qa

# processing de la réponse du llm (afficher les sources)
def process_llm_response(llm_response):
    li = []
    for source in llm_response["source_documents"]:
        li.append(f"Sources:\n\nPage : {source.metadata['page']}. \n Lieu : {source.metadata['source']}. \
              \nContenu : '''{source.page_content}'''")
    li.append(llm_response['answer'])
    return li

def main():
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 📈")

    # variables de session
    if "local_llm" not in st.session_state:
      st.session_state.local_llm = None

    if "qa" not in st.session_state:
      st.session_state.qa = None

    if "messages_qa" not in st.session_state:
        st.session_state.messages_qa = []

    if "embedding" not in st.session_state:
      st.session_state.embedding = None

    # chargement des .pdf qui se trouvent dans le dossier /documents/
    loader = DirectoryLoader('./documents/', glob="./*.pdf", loader_cls=PyPDFLoader)
    ## pour charger un document .txt sans dossier 
    # loader = TextLoader('attention.txt')

    documents = loader.load()
    # avec la fonction load_document, on découpe en morceau le(s) document(s)
    texts = load_document(documents)
    # on affiche la liste des morceaux de texte
    st.write(texts)

    # téléchargement du modèle d'embedding (transformer-encoder)
    st.session_state.embedding = HuggingFaceInstructEmbeddings(model_name="ggrn/e5-small-v2")
    st.write('Le embedding modèle  a été importé avec succès !')

    # création de DB avec Chroma qui contient les embeddings des morceaux de textes
    vectordb = Chroma.from_documents(documents=texts, # morceaux de textes
                                      embedding=st.session_state.embedding, # modèle d'embedding
                                      persist_directory='db') # lieu de stockage

    # choix du LLM 
    name_model = st.selectbox(
        'Choix du LLM HuggingFace',
        ('bigscience/bloomz-560m', 'bigscience/bloomz-1b1', 'bigscience/bloomz-3b'))

    if st.button('Importer'):
      with st.spinner("Loading..."):
        st.session_state.local_llm = load_LLM(name_model)
      st.write('Le modèle a été importé avec succès !')

      # création de la chaine de QA
      st.session_state.qa = qa(st.session_state.local_llm, vectordb)

    # question de l'utilisateur
    with st.container():
        user_input = st.chat_input("Your message: ", key="user_input")

        # si user_input est non nul
        if user_input is not None and user_input != "":
            # ajout du message à la liste
            st.session_state.messages_qa.append({"message": user_input, "is_user": True})

            with st.spinner("Thinking..."):
              # generation de la réponse avec la chaine de QA
              response = st.session_state.qa({"question": user_input})
            # On ajoute à la liste messages_qa : les sources et la réponse (avec process_llm_response)
            for msg in process_llm_response(response):
              st.session_state.messages_qa.append({"message": msg, "is_user": False})

    # afficher les messages
    for i, msg in enumerate(st.session_state.messages_qa):
      if msg['is_user']:
          message(message=msg['message'], is_user=True, key=str(i) + '_user')
      else:
          message(message=msg['message'], is_user=False, key=str(i) + '_ai')

if __name__ == "__main__":
    main()
