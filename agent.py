from langchain.agents import Tool, initialize_agent
from langchain_groq import ChatGroq
import math
import os
from retriver import VectorRetriever
from llm_service import LLMService
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"]=os.environ.get("GROQ_API_KEY")
class MultiAgentSystem:
    def __init__(self):
        # Initialize tools
        self.tools = [
            Tool(
                name="Calculator",
                func=self.calculate,
                description="Useful for mathematical calculations"
            ),
            Tool(
                name="Dictionary",
                func=self.define,
                description="Useful for word definitions"
            )
        ]
        
        # Setup LLM and agent
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.agent = initialize_agent(
            self.tools, 
            self.llm, 
            agent="openai-functions", 
            verbose=True
        )
        
        # Components for RAG
        self.retriever = VectorRetriever()
        self.llm_service = LLMService()
    
    def calculate(self, input_str):
        try:
            return eval(input_str)
        except:
            return "Error in calculation"
            
    def define(self, word):
        # Simple definition lookup (would use API in production)
        definitions = {
            "API": "Application Programming Interface",
            "RAG": "Retrieval-Augmented Generation",
            "LLM": "Large Language Model"
        }
        return definitions.get(word, f"Definition for {word} not found")
    
    def route_query(self, query):
        if any(keyword in query.lower() for keyword in ["calculate", "compute"]):
            return "calculator", self.agent.run(f"Calculator: {query}")
        elif any(keyword in query.lower() for keyword in ["define", "definition"]):
            return "dictionary", self.agent.run(f"Dictionary: {query}")
        else:
            # RAG pipeline
            context = self.retriever.retrieve(query)
            context_text = "\n".join([doc.page_content for doc in context])
            answer = self.llm_service.generate_answer(context_text, query)
            return "rag", answer, context