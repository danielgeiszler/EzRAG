# EzRag.py
import os
from dotenv import load_dotenv
import nltk
import gradio as gr
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

# Initialize environment
load_dotenv()
nltk.download('punkt')
nltk.download('punkt_tab')

class DeepSeekRunnable(Runnable):
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
        )
    
    def invoke(self, input: dict, config: dict = None, **kwargs):
        """Handle LangChain compatibility"""
        try:
            query = input.get("query") if isinstance(input, dict) else str(input)
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": query}],
                temperature=0.3,
                **kwargs
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error: {str(e)}"

def initialize_rag():
    try:
        # 1. Load and split documents
        loader = DirectoryLoader("./data", glob="**/*.txt")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)

        # 2. Create vector store with updated embeddings
        embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_db = FAISS.from_documents(chunks, embed_model)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})

        # 3. Create RAG chain
        template = """Use the context below to answer. If unsure, say "I don't know". 
        
        Context: {context}
        Question: {question}
        Answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=DeepSeekRunnable(),
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
            input_key="query"
        ).with_config(run_name="DeepSeekRAG")
        
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        exit(1)


# Gradio interface
def ask(question):
    try:
        response = rag_chain.invoke({"query": question})
        return response['result']
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Initialize system
    rag_chain = initialize_rag()

    gr.Interface(
        fn=ask,
        inputs=gr.Textbox(label="Question"),
        outputs=gr.Textbox(label="Answer"),
        title="DeepSeek Document Assistant"
    ).launch()
