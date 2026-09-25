from pypdf import PdfReader

#doc_path = "documents/RK.pdf"

def create_chunks(doc_path, chunk_size = 500, overlap_size = 100 ):
    reader = PdfReader(doc_path)
    full_text = ''

    for page in reader.pages:
        text = page.extract_text()
        full_text = full_text + text + '\n'
    #print(len(full_text))

    ids, chunks, metadatas, start, chunk_id = [], [], [], 0, 1

    while start< len(full_text):
        end = start+chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk)
        ids.append(f'{doc_path}_{chunk_id}')
        metadatas.append({"doc_name":doc_path, "version":1})
        chunk_id = chunk_id + 1
        start=end - overlap_size
    return chunks, ids, metadatas

'''
chunks, ids, metadatas = create_chunks(doc_path)
print(f'len(chunks): {len(chunks)}')
print(chunks[1])
print(ids[4])
print(metadatas[4])
'''
