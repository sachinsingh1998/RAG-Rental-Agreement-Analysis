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

#print(collection.get(ids=['documents/Ana.pdf_10']))

while True:

    document_name = input("Enter Document: ")

    if document_name == "exit":
        break

    query = input(f"Enter your query according to document {document_name}: ")

    query_embedding = get_embeddings(query, client)

    related_docs = collection.query(
        query_embeddings = query_embedding,
        n_results=5,
        where = {
            "doc_name":f"documents/{document_name}"
        }
    )

    #print(related_docs)
    #print(related_docs["documents"])

    print(related_docs["ids"][0])
    for doc in related_docs["documents"][0]:
        print("---")
        print(doc[:220])

    context = related_docs["documents"][0]

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
    print(f"Answer to your query {query} is: \n")
    print(response.output_text)
    

print("Goodbye")