import numpy as np


def text_embedding(model, text):
    embeddings = model.encode(text)
    return embeddings


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)

    return dot_product / (norm_a * norm_b)


def calc_similarity(model, text, text2):
    embeddings_1 = text_embedding(model, text)
    embeddings_2 = text_embedding(model, text2)
    similarity = cosine_similarity(embeddings_1, embeddings_2)
    return similarity
