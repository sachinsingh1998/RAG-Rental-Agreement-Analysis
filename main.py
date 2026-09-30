from ingestion import document_ingestion
import os
from retrieval import rag

folder = 'documents'
files = os.listdir(folder)


#ingestion phase
#created chunks for all files in folder
#can skip if already done
#document_ingestion(folder)

#retrieval phase -- user input query
while True:

    document_name = input(f"Enter a Document from {files}: ")

    if document_name == "exit":
        break
    
    if document_name not in files:
        print("Enter the correct file name mate !")
        continue

    query = input(f"Enter your query according to document {document_name}: ")

    response = rag(query,document_name)
    
    print(f"\nAnswer to your query {query} is: \n")
    print(response)
        
print("Goodbye")
