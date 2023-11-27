import os
import torch
import transformers
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import streamlit as st
from streamlit_chat import message

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory

# Charger le local_LLM
def load_LLM(name_model):
  tokenizer = AutoTokenizer.from_pretrained(name_model)
  model = AutoModelForCausalLM.from_pretrained(name_model,
                                                load_in_8bit=True, # charger en 8bit avec bitsandbytes
                                                device_map='auto', # utiliser les GPU configuration auto
                                                torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True)
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

# création de la chaine
def chaine(local_llm):
  # prompt utilisé
  template = """You are a AI having a conversation with a human.

  {chat_history}
  Human: {human_input}
  AI:"""

  prompt = PromptTemplate(
                          input_variables=["chat_history", "human_input"],
                          template=template
                          )
  # création de la mémoire
  memory = ConversationBufferMemory(memory_key="chat_history")
  # chaine avec mémoire
  llm_chain = LLMChain(
                      llm=local_llm,
                      prompt=prompt,
                      verbose=True, # pour vérifier le prompt des inférences dans le terminal
                      memory=memory,
                      )
  return llm_chain

def main():
    st.set_page_config(page_title="ChatBot 🤖")
    st.markdown("# Mon Chatbot Local ! 🤖")
    st.markdown("## Cette application - Chatbot utilise LangChain & Streamlit")
    st.markdown("Ce chatbot admet un :red[prompt personnalisable] et a de la **:blue[mémoire]** !")

    st.sidebar.success('Choix du UC')

    # variables de session qui sont gardées en mémoire : le llm et les messages
    if "llm_chain" not in st.session_state:
      st.session_state.llm_chain = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # choix du modèle
    name_model = st.selectbox(
        'Choix du LLM HuggingFace',
        ('bigscience/bloomz-560m', 'bigscience/bloomz-1b1', 'bigscience/bloomz-3b'))

    if st.button('Importer'):
      with st.spinner("Import en cours..."):
        local_llm = load_LLM(name_model)
      st.session_state.llm_chain = chaine(local_llm)

      st.write('Le modèle a été importé avec succès !')

    with st.container():
        # message de l'utilisateur dans user_input
        user_input = st.chat_input("Your message: ", key="user_input")

        # si user_input est non vide :
        if user_input is not None and user_input != "":
            # ajout de ce message dans la liste messages
            st.session_state.messages.append({"message": user_input, "is_user": True})

            with st.spinner("Thinking..."):
              # inférence au modèle avec llm_chain.predict
              response = st.session_state.llm_chain.predict(human_input=user_input)
            # ajout de la réponse dans la liste messages
            st.session_state.messages.append({"message": response, "is_user": False})

    # affichage de l'historique des messages
    for i, msg in enumerate(st.session_state.messages):
      # numeros pairs : utilisateur
      if i % 2 == 0:
          message(message=msg['message'], is_user=True, key=str(i) + '_user')
      else:
          message(message=msg['message'], is_user=False, key=str(i) + '_ai')

if __name__ == "__main__":
    main()
