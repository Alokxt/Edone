import faiss 
import pickle
from sentence_transformers import SentenceTransformer
import os 

index_path = os.path.join(os.path.dirname(__file__), "IBPS_embedds.faiss")

def get_transformer():
    transformer = SentenceTransformer('all-MiniLM-L6-v2')
    return transformer

def get_index():
    index = faiss.read_index(index_path)
    return index
def get_chunks():
    chunk_path = os.path.join(os.path.dirname(__file__), "chunks.pkl")

    with open(chunk_path,'rb') as f:
        chunks = pickle.load(f)
    return chunks
