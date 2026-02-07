from bytez import Bytez 
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
import os 
load_dotenv()
bytez_api_key = os.environ["BYTEZ_API_KEY"]
class BytezEmbeddings(Embeddings):
  def __init__(self,api_key,model_name):
    self.api_key = api_key
    self.sdk = Bytez(api_key)
    self.model = self.sdk.model(model_name)

  def _embed(self,text):
    result = self.model.run(text)
    if result.error is not None:
      raise RuntimeError(result.error)
    return result.output

  def embed_documents(self, texts: list[str]) -> list[list[float]]:

     return [self._embed(f"search_document:{c}") for c in texts]
  def embed_query(self,text):
    return self._embed(f"search_query:{text}")
  
def get_embeddings():
  return BytezEmbeddings(api_key=bytez_api_key,model_name="nomic-ai/nomic-embed-text-v1.5")