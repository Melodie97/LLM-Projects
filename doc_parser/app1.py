import document_processor
import embeddings
import model
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel


class Docs(BaseModel):
    upload_document: Optional[object]

app = FastAPI()

@app.get("/")
async def root():
    return "Welcome to Doc parser"


@app.post("/model")
def main(uploaded_docs: Docs, model_name="Mistral7B"):
    if st.button("Compute"):
        with st.spinner("Training model on your document(s)"):
            raw_text = document_processor.get_pdf_text(uploaded_docs)
            text_chunks = document_processor.get_text_chunks(raw_text)
            vector_database = embeddings.get_vector_database(text_chunks, model_name)
            st.session_state.conversation = model.chatbot(vector_database, model_name)
            st.write('Training Completed')
            
