from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from Products.config import Config
import re
import unicodedata



class RAGChainBuilder:
    def __init__(self,vector_store):
        self.vector_store = vector_store
        self.model = ChatGroq(model=Config.RAG_MODEL, temperature=0.5)
        self.history_store = {}
    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]
    
    
    # @staticmethod
    def normalize_query(self,text):
        if isinstance(text, list):
            text = " ".join(text)  
        return unicodedata.normalize("NFKC", text.strip())

    
    def build_chain(self):
        retriever= self.vector_store.as_retriever(search_kwargs={"k": 3})
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and user question, rewrite it as a standalone question."),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  
        ])

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You're an e-commerce bot answering product-related queries using reviews and titles.
                          Stick to context. Be concise and helpful.\n\nCONTEXT:\n{context}\n\nQUESTION: {input}"""),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  
        ])
        
        history_aware_retriever=create_history_aware_retriever(
            self.model,retriever,context_prompt
        )
        
        question_answer_chain=create_stuff_documents_chain(
            self.model, qa_prompt
        )
        normalize_input = RunnableLambda(lambda x: {
        "input": self.normalize_query(x["input"]),
        "chat_history": x.get("chat_history", []), 
        "context": x.get("context", "") 
        })
     
        rag_chain=create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )
        
        return RunnableWithMessageHistory(
            normalize_input |rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
  