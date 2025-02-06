# EzRag.py
import os
from dotenv import load_dotenv
import nltk
import gradio as gr
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from pathlib import Path
import argparse

# Initialize environment
load_dotenv()
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


class DeepSeekRunnable(Runnable):
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
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

class OpenAIRunnable(Runnable):
    def __init__(self, temperature = 0.3):
        self.temperature = temperature
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),  # Different environment variable
            # base_url is not needed as it defaults to OpenAI's official endpoint
        )

    def invoke(self, input: dict, config: dict = None, **kwargs):
        try:
            # Maintain the same input handling as DeepSeek implementation
            query = input.get("query", "") if isinstance(input, dict) else str(input)

            response = self.client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[{"role": "user", "content": query}],
                temperature=self.temperature,
                **kwargs
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"Error: {str(e)}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run EzRAG with configurable service.",
        epilog="Example: python fragrag.py --service openai"
    )
    parser.add_argument(
        "--service",
        choices=["openai", "deepseek"],
        default="openai",
        help="The service to use: 'openai' or 'deepseek' (default: openai)."
    )
    return vars(parser.parse_args())

def safe_load_documents(directory: str):
    documents = []
    errors = []

    # Gather all .txt and .pdf files, excluding hidden files/directories
    supported_extensions = ['.txt', '.pdf', '.docx']
    files = [
        f for f in Path(directory).rglob("*")
        if f.suffix.lower() in supported_extensions and not any(part.startswith(".") for part in f.parts)
    ]
    print(f"Found {len(files)} files to process")

    for file in files:
        try:
            if file.suffix.lower() == ".txt":
                loader = TextLoader(str(file), autodetect_encoding=True)
            elif file.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file))
            elif file.suffix.lower() == ".docx":
                loader = UnstructuredWordDocumentLoader(str(file))
            else:
                # This should not happen as we've filtered supported extensions
                raise ValueError("Unsupported file extension")

            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            errors.append((str(file), str(e)))
            continue

    print(f"Success: {len(documents)} docs | Failed: {len(errors)}")
    if errors:
        print("First 5 errors:")
        for file, error in errors[:5]:
            print(f" - {file}: {error}")

    return documents


def initialize_rag(llm):
    try:
        # 1. Load and split documents
        data_dir = Path("./data").resolve()
        documents = safe_load_documents(data_dir)

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
            llm=llm,
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
    # Get args
    config = parse_args()
    service = config.get("service")

    # Choose the runnable based on service
    if service.lower() == "openai":
        llm = OpenAIRunnable()
    elif service.lower() == "deepseek":
        llm = DeepSeekRunnable()
    else:
        raise ValueError(f"Unsupported service: {service}")

    # Initialize system
    rag_chain = initialize_rag(llm)

    gr.Interface(
        fn=ask,
        inputs=gr.Textbox(label="Question"),
        outputs=gr.Textbox(label="Answer"),
        title="Document Assistant"
    ).launch()
