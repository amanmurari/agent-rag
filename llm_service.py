import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"]=os.environ.get("GROQ_API_KEY")
class LLMService:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        
        self.prompt_template = PromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the question.
Context: {context}
Question: {question}
Answer:
""")
        
    def generate_answer(self, context, question):
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        return self.llm.predict(prompt)