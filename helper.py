def get_embeddings(text,client):
    response = client.embeddings.create(
        model='text-embedding-3-small',
        input = text,
        dimensions = 300
    )
    return response.data[0].embedding

def get_embeddings_in_batch(texts,client):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
        dimensions = 300
    )

    embeds = []

    for item in response.data:
        embedding = item.embedding
        embeds.append(embedding)
    return embeds