import os
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv


def ada_embedding(text: str, mode: str = "search"):
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
    os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
    deployment_name = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT_NAME")
    
    embeddings = OpenAIEmbeddings(model=deployment_name)
    
    if mode == "search": result = embeddings.embed_documents([text])
    elif mode == "query": result = embeddings.embed_query(text)
    else: raise ValueError("mode must be either search or query")
    return result
