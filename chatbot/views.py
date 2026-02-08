import re 
import json 
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from samadhanai.settings import DEEPSEEK_API_KEY
from openai import OpenAI
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
import os, shutil, uuid, time, tempfile
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.core.cache import cache
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from dotenv import load_dotenv
from upstash_redis import Redis
from bytez import Bytez
from functools import lru_cache
from .embds import get_embeddings

load_dotenv()
bytez_api_key = os.environ["BYTEZ_API_KEY"]
redis = Redis(url="https://well-mayfly-23243.upstash.io", token=os.getenv('REDIS_TOKEN'))

def get_vectorstores():
    from langchain_community.vectorstores import FAISS
    return FAISS

def get_textsplitter():
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

def get_chatmodel():
    from langchain_openai import  ChatOpenAI
    model = ChatOpenAI(model="tngtech/deepseek-r1t2-chimera:free",api_key=os.getenv("OPENROUTER_API_KEY"),   
    base_url="https://openrouter.ai/api/v1" ,temperature=0.2)
    return model 

def get_prompttemp():
    from langchain_core.prompts import PromptTemplate
    return PromptTemplate




client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=DEEPSEEK_API_KEY,
)


@lru_cache()
def get_vectorr():
   FAISS = get_vectorstores()
   vectorR = FAISS.load_local( "/data2/rag_index2", get_embeddings(), allow_dangerous_deserialization=True )
   return vectorR


def generate_response(query,context,metadata):
    PromptTemplate = get_prompttemp()
    prompt = PromptTemplate(
        template="""
    You are an academic assistant for undergraduate core engineering subjects.

Explain topics accurately using standard engineering knowledge.
The provided context is only for syllabus scope alignment.

Rules:
- Stay strictly within core engineering subjects.
- Do not introduce unrelated domains.
- If the question is outside the subject, say so clearly.
- Use a formal, textbook-style explanation.

context:
{context}

Metadata:
{metadata}

User query:
{query}


Explaination should be Detailed and correct. Do not hallucinate.

Mention standard textbooks or academic sources (no URLs).

\n

        """,
        input_variables=['context','query','metadata'],
       
    )
    model = get_chatmodel()
    
    chain = prompt | model 
   
    

    raw = chain.invoke({'context':context,'query':query,'metadata':metadata})

    return raw
    

    


@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def home(request):
    try:
        data = json.loads(request.body)
        query = data.get('query')
      
        if query is None:
            return Response({'success':False,"Message":"Ask something"},status=400)

        vectorR = get_vectorr()
        docs = vectorR.similarity_search(
            query,
            k=2
        )

        

        context = " ".join(x.page_content for x in docs )
        metadata = docs[0].metadata
      
      
        ans  = generate_response(query,context,metadata)
        
       
        
        return Response({"answer":ans},status=201)
    except Exception as e:
        return Response({"error":f"something went wrong {e}"},status=500)
    
def generate_chat_response(context,query):
    PromptTemplate = get_prompttemp()
    video_rag_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    You are an AI assistant analyzing the content of a video.

    You are given VIDEO CONTEXT extracted from the video transcript using a retrieval system.
    This context is the primary source of truth.

    VIDEO CONTEXT:
    ----------------
    {context}
    ----------------

    USER QUESTION:
    {question}

    Instructions:

    1. Base your answer primarily on the provided video context.
    2. Explain concepts using details, examples, and explanations from the context.
    3. You may use general domain knowledge only to clarify or briefly expand ideas already present.
    4. Do NOT introduce unrelated topics.
    5. Do NOT contradict or override the video content.
    6. If the context does not contain enough information to answer fully, say so clearly.

    Now provide a clear, relevant, and grounded explanation.
    """
    )
    model = get_chatmodel()
   
    chain = video_rag_prompt | model 
    result = chain.invoke({'context':context,'question':query})
    return result

    
from urllib.parse import urlparse, parse_qs
def extract_video_id(url):
    parsed = urlparse(url)

    if parsed.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed.query).get("v", [None])[0]

    if parsed.hostname == "youtu.be":
        return parsed.path[1:]

    return None
    
 

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def youtubevideo(request):
    try:
        data = json.loads(request.body)
        video_url = data.get('video_url','')
        lang = data.get('language')
        user = user 

        redis_key = f"yt_session:{user.id}"

        old = redis.get(redis_key)
        if old:
            if isinstance(old, str):
                    old = json.loads(old)
            
            shutil.rmtree(old["vector_path"], ignore_errors=True)
            redis.delete(redis_key)
        session_id = str(uuid.uuid4())
        session_dir = tempfile.mkdtemp(prefix=f"yt_{session_id}_")
        lan = 'en'
        if lang == "Hindi":
            lan = 'hi'
        
        
        
        video_id = extract_video_id(video_url)
        yt = YouTubeTranscriptApi()
        transcript_list = yt.fetch(video_id=video_id, languages=[lan])
        if not transcript_list:
            return Response({'success':False,"Message":"Could not fetch the video , see if the video is available"})
        
        transcript = " ".join(t.text for t in transcript_list)

        splitter = get_textsplitter()
        chunks = splitter.create_documents([transcript])

        
        FAISS = get_vectorstores()
        vectorstore = FAISS.from_documents(
            chunks,
            embedding=get_embeddings()
        )

        vectorstore.save_local(session_dir)

        redis.set(
            redis_key,
            {
                "session_id":session_id,
                "vector_path":session_dir,
                "video_id":video_id,

            },
            ex=30 * 60
        )

        return Response({'success':True,"Message":"Stored the session details","session_id":session_id})
    except Exception as e:
        return Response({'success':False,"Message":f"An error occurred {e}"})



  
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def chatvideo(request):
    try:
       data = json.loads(request.body)
       session_id = data['session_id']
       usr = request.user
       query = data['query']



       redis_key = f"yt_session:{usr.id}"
       old = redis.get(redis_key)
       if not old:
           return Response({'success':False,"Message":"No previous session found/session timeout"})
       if isinstance(old, str):
            old = json.loads(old)
       
       vector_path = old["vector_path"]
       if session_id != old['session_id']:
           return Response({'success':False,"Message":"Different sessions ,previous session expired"})
       FAISS = get_vectorstores()
       vector_store = FAISS.load_local(vector_path,embeddings=get_embeddings(),allow_dangerous_deserialization=True)
       retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

       ret = retriever.invoke(query)

       context = " ".join(t.page_content for t in ret)

       ans = generate_chat_response(context,query)

       return Response({'success':True,"data":ans})
    
    except Exception as e:
        return Response({'success':False,"Message":f"An error occurred {e}"})

def clean_llm_json(text):
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)
  
@api_view(["POST"])
@permission_classes([AllowAny])
def quiz_generator(request):
    try:
        data = json.loads(request.body)
        topics = data.get('topics','')
     
        if len(topics) == 0:
            return Response({"success":False,"Message":"Give some topics"},status=400)
        vectorR = get_vectorr()
        docs = vectorR.similarity_search(
            topics,
            k=2
        )
        num_ques = data.get('num_ques')
        num_ques = int(num_ques)
        print(topics)
        print(num_ques)

        context = " ".join(x.page_content for x in docs )
        metadata = docs[0].metadata
      
      
        ans  = generate_response(topics,context,metadata)
        PromptTemplate = get_prompttemp()
        temp = PromptTemplate(
            template="""
You are an AI exam paper generator for an Engineering-level technical assessment.

You will be given STUDY CONTENT below.
This content is the ONLY authoritative source you may use to create questions.

Rules you must follow strictly:

• All questions must be based on the provided content
• You may ask on closely related concepts implied by the content, but nothing outside its scope
• Difficulty level must be MEDIUM to HARD (conceptual + application based, not trivial recall)
• Do NOT introduce topics not present in the content
• Avoid vague or opinion-based questions
• Ensure each question has one clear correct answer

Test requirements:

• Total questions: {num_ques}
• Mix of:
– Conceptual understanding
– Algorithm/process reasoning
– Edge cases or practical implications

For each question provide:

The question

The correct answer

A short explanation justifying the answer



STUDY CONTENT:
{content}

If any part of the content is insufficient to generate the meaningful medium-hard questions, focus on depth rather than inventing new topics.
You should Strictly return the Reponse in this Format only .
Return ONLY valid JSON. Do not include markdown, code fences, or any text outside the JSON object.
{{
                "questions": [
                    {{
                    "question": "Question text here",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "answer": "Correct Option (e.g., B)"
                    "explanation": "Why the correct option is correct."
                    }},
                    {{
                    "question": "...",
                    "options": ["...", "...", "...", "..."],
                    "answer": "...",
                    "explanation": "Why the correct option is correct."
                    }}
                ]
                }}
""",
input_variables=['num_ques','content'],
        )
        model = get_chatmodel()
      
        chain = temp | model 

   
    

        raw = chain.invoke({'content':ans,'num_ques':num_ques})
        raw = clean_llm_json(raw)
     

        return Response({"success":True,"content":raw})
    except Exception as e:
        return Response({"success": False, "error": str(e)},status=500)



