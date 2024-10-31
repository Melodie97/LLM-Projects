import model
import time
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from model import table_info


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_state = None

class UserInput(BaseModel):
    user_input: str

# class DBName(BaseModel):
#     database: str


@app.get("/")
async def root():
    return "Welcome to Text_to_SQL"


@app.post("/chat/")
async def conversation(input: UserInput):
    try:
        print(input.user_input)
        print("================================================================================")
        start = time.time() 
        response = {"success":True,
                    "chat":model.chatbot(input.user_input)}
        # response = session_state({'question': input.user_input})
        end = time.time()
        print(f'Time taken to run {end-start}secs')
    except Exception:
        response = {"detail":"Please try to rephrase your question to be as specific as possible",
                    "info":f"Your database has the following tables {table_info}. Please ask questions about these tables."}
    return response

