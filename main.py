from chunking import create_chunks

doc_path = "documents/RK.pdf"

chunks, ids, metadatas = create_chunks(doc_path)

'''
print(f'len(chunks): {len(chunks)}')
print(chunks[1])
print(ids[4])
print(metadatas[4])
'''
