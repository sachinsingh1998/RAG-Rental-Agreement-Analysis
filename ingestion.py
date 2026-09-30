import chromadb 
from openai import OpenAI
from dotenv import load_dotenv
from chunking import create_chunks
from helper import get_embeddings,get_embeddings_in_batch
import os

#folder = 'documents'

load_dotenv()
client = OpenAI()


'''
embedding = get_embeddings("I want to have sex with an australian girl", client)
print(embedding)
'''
def document_ingestion(folder):
    db_client = chromadb.PersistentClient("./chroma_db")
    collection = db_client.get_or_create_collection(name = "rental_agreements")

    files = os.listdir(folder)
    print(files)

    for file in files:
        doc_path = f'{folder}/{file}'
        chunks, ids, metadatas = create_chunks(doc_path)
        embeddings = get_embeddings_in_batch(chunks,client)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents = chunks,
            metadatas = metadatas
        )
    
