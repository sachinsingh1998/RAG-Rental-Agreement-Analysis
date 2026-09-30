import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from helper import get_embeddings
import os

load_dotenv()
my_model = os.getenv('MODEL_NAME')

client = OpenAI()

db_client = chromadb.PersistentClient("./chroma_db")
collection = db_client.get_or_create_collection(name = "rental_agreements")

#function to retrieve top_k related documents based on user query
def get_retrieved_docs(query, document_name, top_k=5):
    
    query_embedding = get_embeddings(query, client)
    
    related_docs = collection.query(
        query_embeddings = query_embedding,
        n_results=top_k,
        where = {
            "doc_name":f"documents/{document_name}"
        }
    )
    
        #print(related_docs)
        #print(related_docs["documents"])
    context = related_docs["documents"][0]

    print("Related docs are: ")
    print(related_docs["ids"][0])
    for doc in context:
        print("---")
        print(doc[:220])
    return context


folder = 'documents'
files = os.listdir(folder)

def rag(question, document_name, top_k = 5):
    #print("Chudh gaye !")
    context = get_retrieved_docs(query, document_name)
        
    prompt = f"""
    You are a rental agreement lawyer based of Sydney, Australia who scruntize rental agreement 
    documents from client and answers their questions.
    Based on the given context answer user question: 
    context : {context}
    user_question : {query}

    Rules:
    1.Give direct and natural answers.
    2. IF you don't have adequate context for question, dont make up the answer, just say I dont know.
    """

    #print(prompt)

    response = client.responses.create(
        input=prompt,
        model=my_model
    )
    return response.output_text
    
    


#print(collection.get(ids=['documents/Ana.pdf_10']))

while True:

    document_name = input(f"Enter a Document from {files}: ")

    if document_name == "exit":
        break
    
    if document_name not in files:
        print("Enter the correct file name mate !")
        continue

    query = input(f"Enter your query according to document {document_name}: ")

    response = rag(query,document_name)
    
    print(f"Answer to your query {query} is: \n")
    print(response)
        

    

print("Goodbye")