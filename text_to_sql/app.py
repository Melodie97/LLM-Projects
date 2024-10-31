import streamlit as st
import model
import time
from html_chat_template import css, bot_template, user_template
from html_template import input_text_style, header_style



def submit():
    st.session_state.submit_input = True
    st.session_state.user_question = st.session_state.user_input 
    st.session_state.user_input = ""
    #clear_input_field()       
    
    
def handle_userinput(user_question):
    st.session_state.conversation = model.chatbot(user_question)
    response = st.session_state.conversation
    st.session_state.chat_history = response
    print(response)

    # for i, message in enumerate(st.session_state.chat_history):
    #     if i % 2 == 0:
    st.write(user_template.replace(
    "{{MSG}}", user_question), unsafe_allow_html=True)
        # else:
    st.write(bot_template.replace(
    "{{MSG}}", st.session_state.chat_history), unsafe_allow_html=True)
            
                                       
def main():
    start_time =  time.time()
    st.set_page_config(page_title="Retrieve any SQL Query")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None   
          
    if "submit_input" not in st.session_state:
        st.session_state.submit_input = False
        
    if "user_question" not in st.session_state:   
        st.session_state.user_question = "" 
               
    
    header_container = st.container()
    chat_container = st.container()
    input_container = st.container()
    
    
    with header_container:
        header_styles = header_style.get('style')
        st.markdown(header_styles, unsafe_allow_html=True)
        st.header("App to Retrieve any SQL Query")
        
        
    
    with input_container:
        text_column, button_column = st.columns([5, 1])
        input_text_styles = input_text_style.get('style')
        st.markdown(input_text_styles, unsafe_allow_html=True)
        with text_column:            
            user_input = st.text_input("", key='user_input', on_change=submit, placeholder="What would you like to know about your database?")
        st.markdown(input_text_styles, unsafe_allow_html=True)
        with button_column:
            st.markdown(input_text_styles, unsafe_allow_html=True)
            st.write("")
            send_button = st.button('send', key='send_button')
    
        if send_button or st.session_state.submit_input:
            print('yes')
            print (st.session_state.user_question) 
            if st.session_state.user_question != "":
                print (st.session_state.user_question)
                with chat_container:
                    handle_userinput(st.session_state.user_question)
    
    

    # with st.sidebar:
    #     model_name = st.selectbox("Select your prefered LLM", ["Mistral7B"])
    #     if st.button("Compute"):
    #         with st.spinner("Training model on your document(s)"):
    #             raw_text = document_processor.get_pdf_text(uploaded_docs)
    #             text_chunks = document_processor.get_text_chunks(raw_text)
    #             vector_database = embeddings.get_vector_database(text_chunks, model_name)
    #             st.session_state.conversation = model.chatbot(vector_database, model_name)
    #             end_time = time.time()
    #             time_taken = end_time-start_time
    #             st.write('Training Completed')
    #             st.write(f'Time taken to run the model {time_taken}')
                

if __name__ == '__main__':
    main()
    
        