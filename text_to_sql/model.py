# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_community.llms import LlamaCpp
# from langchain_community.llms import CTransformers, LlamaCpp
# from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
# from langchain.chains import create_history_aware_retriever
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
import sqlite3
# from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, inspect
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
import pyodbc
import pymssql
from urllib.parse import quote

from langchain_core.output_parsers import StrOutputParser
#OutputParser that parses LLMResult into the top likely string.


# from langchain.schema.runnable import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain_core.runnables import RunnablePassthrough



conn = pyodbc.connect('DRIVER={SQL Server};SERVER=XXX;DATABASE=FeatureStore;UID=XX;PWD=XXX')
cursor = conn.cursor()
table_schema = """
select * from information_schema.columns
"""
conn.commit()
cursor.close()
conn.close()

user_id = "XXX"
password = quote("XX")
server = "XXX"
database = "FeatureStore"

uri = f"mssql+pymssql://{user_id}:{password}@{server}:1433/{database}"
db = SQLDatabase.from_uri(uri)

# db = SQLDatabase.from_uri("sqlite:///work.db")
dialect = db.dialect
table_info = db.get_usable_table_names()
print(db.get_usable_table_names())
# engine = create_engine('sqlite:///employee.db')
# insp = inspect(engine)
# print(insp.get_table_names())

# template = """Question: {question}

# Answer: Let's work this out in a step by step way to be sure we have the right answer."""

# prompt = PromptTemplate.from_template(template)

# Callbacks support token-wise streaming
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

def load_mistral_pipeline():
    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    llm = LlamaCpp(
        model_path= "./Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        temperature=0,
        top_p=3,
        n_ctx=30088,
        max_new_tokens=256,
        context_window=3900,
        generate_kwargs={},
        streaming=True,
        verbose=True,
        n_gpu_layers=0
    )
    return llm

# def load_mistral_pipeline():
#     llm = Llama(
#     model_path="./Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_M.gguf",  # Download the model file first
#     chat_format="llama-2",
#     n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
#     n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
#     n_gpu_layers=0         # The number of layers to offload to GPU, if you have GPU acceleration available
#     )
#     return llm

 
# def set_custom_prompt():
#     """ prompt template for custom prompt"""  
#     # prompt = PromptTemplate(template=custom_prompt, input_variables=['context', 'question'])
#     # return prompt

#     contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", custom_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
#     )
#     return contextualize_q_prompt


def write_query(): 
    llm_models = {
        'Mistral7B': load_mistral_pipeline()
    }
    llm = llm_models.get('Mistral7B')
    
    # sql_prompt_t = """
    #     Given an input question, create a syntactically correct query to run.
    #     Only use the following tables:

    #     {table_info}.

    #     {top_k}

    #     Question: {input}'''
        
    # """
    # sql_prompt = PromptTemplate.from_template(sql_prompt_t)
    chain = create_sql_query_chain(llm, db, k=2)

    # print(chain.invoke({"question": "how many employees are there"}))
    return llm, chain


def execute_query():
    execute_query = QuerySQLDataBaseTool(db=db)
    query = write_query()[1]
    chain = query | execute_query
    return chain

answer_prompt =\
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question in a crisp manner.
        If you don't know the correct answer, just say that you don't know, don't try to make up an answer.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """

def set_custom_prompt():
    """ prompt template for custom prompt"""  
    prompt = PromptTemplate(template=answer_prompt, input_variables=['question', 'query', 'result'])
    return prompt


def chatbot(question):
    print("Starting.................")
    llm, query = write_query()
    print("write query chain gotten")
    query = query.invoke({"question": question})
    print("!!!!!!!!!!!!!!!!!!!!Write Query Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    result = execute_query()
    print("execute query chain gotten")
    result = result.invoke({"question": question})
    print("!!!!!!!!!!!!!!!!!!!!Execute Query Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    prompt = set_custom_prompt()
    print(query)
    print("""
        ======================================================================================
        ======================================================================================
    """)
    print(result)
    
    # llm, w_query = write_query()
    # # e_query = execute_query()
    # answer = answer_prompt | llm | StrOutputParser()
    
    # chain = (
    #     RunnablePassthrough.assign(query=w_query).assign(
    #     result=itemgetter("query") | execute_query
    #     )
    #     | answer
    #     )
    # chain = RunnablePassthrough..assign(query=write_query).assign(
    # chat_history=RunnableLambda(readonly_memory.load_memory_variables) | itemgetter("chat_history")
    # )
    llm_chain = prompt | llm
    response = llm_chain.invoke({"question": question, "query": query, "result": result})
    return response


# question = "how many employees are there?"
# chain1 = chatbot(question)
# print(chain1)

#retrieve query from db
def read_sql(sql, db):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    conn.commit()
    conn.close()
    for row in rows:
        print(row)
    return rows

# custom_prompt=[
#     f"""
#     [INST]
#     You are an SQL expert.
#     Please help to generate an SQL query to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions. 

#     ==Tables:
#     {table_schema}
    
#     Use the following example to retreive sql queries.
#     Example 1 - How many entries of records are present?, 
#     the SQL command will be something like this: SELECT COUNT(*) FROM STUDENT ;
#     Example 2 - Tell me all the students studying in Data Science class?, 
#     the SQL command will be something like this: SELECT * FROM STUDENT where CLASS="Data Science"; 
#     also the sql code should not have ``` in the beginning or end of an sql word in the output
#     [/INST]
#     """
# ]