
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace , HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from upstash_redis import Redis


load_dotenv()
'''redis = Redis(url="https://well-mayfly-23243.upstash.io", token=os.getenv('REDIS_TOKEN'))

parser = StrOutputParser()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=DEEPSEEK_API_KEY,
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorR = FAISS.load_local(
            "C:\\Users\\Nimisha Manawat\\OneDrive\\Desktop\\StudyBuddy\\samadhanai\\chatbot\\data2\\rag_index",
            embeddings,
            allow_dangerous_deserialization=True
        )


model = ChatOpenAI(model="tngtech/deepseek-r1t2-chimera:free",api_key=os.getenv("OPENROUTER_API_KEY"),   
    base_url="https://openrouter.ai/api/v1" ,temperature=0.2)

def generate_response(query,context,metadata):
    
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
 
    chain = prompt | model | parser 
   
    

    raw = chain.invoke({'context':context,'query':query,'metadata':metadata})

    return raw
    

    


@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def home(request):
    try:
        
        query = request.data.get('query')
      
        if query is None:
            return Response({'success':False,"Message":"Ask something"},status=500)

         
        docs = vectorR.similarity_search(
            query,
            k=2
        )

        

        context = " ".join(x.page_content for x in docs )
        metadata = docs[0].metadata
      
      
        ans  = generate_response(query,context,metadata)
       
        
        return Response({"answer":ans},status=201)
    except Exception as e:
        return Response({"error":f"something went wrong {e}"},status=400)
    
def generate_chat_response(context,query):
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

    chain = video_rag_prompt | model | parser 
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
    '''
    
@api_view(["POST"])
@permission_classes([AllowAny])
def home(request):
    return Response({'success':True,"Message":"Hello"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def youtubevideo(request):
    try:
        '''video_url = request.data.get('video_url','')
        lang = request.data.get('language')
        user = request.user 

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

        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        chunks = splitter.create_documents([transcript])

        

        vectorstore = FAISS.from_documents(
            chunks,
            embedding=embeddings
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
        )'''

        return Response({'success':True,"Message":"Stored the session details","session_id":1})
    except Exception as e:
        return Response({'success':False,"Message":f"An error occurred {e}"})



  
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def chatvideo(request):
    try:
       '''session_id = request.data['session_id']
       usr = request.user
       query = request.data['query']



       redis_key = f"yt_session:{usr.id}"
       old = redis.get(redis_key)
       if not old:
           return Response({'success':False,"Message":"No previous session found/session timeout"})
       if isinstance(old, str):
            old = json.loads(old)
       
       vector_path = old["vector_path"]
       if session_id != old['session_id']:
           return Response({'success':False,"Message":"Different sessions ,previous session expired"})
       
       vector_store = FAISS.load_local(vector_path,embeddings=embeddings,allow_dangerous_deserialization=True)
       retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

       ret = retriever.invoke(query)

       context = " ".join(t.page_content for t in ret)

       ans = generate_chat_response(context,query)'''

       return Response({'success':True,"data":1})
    
    except Exception as e:
        return Response({'success':False,"Message":f"An error occurred {e}"})

def clean_llm_json(text):
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)
  
@api_view(["POST"])
@permission_classes([AllowAny])
def quiz_generator(request):
    try:
        '''topics = request.data.get('topics')
        if len(topics) == 0:
            return Response({"success":False,"Message":"Give some topics"})
        docs = vectorR.similarity_search(
            topics,
            k=2
        )
        num_ques = request.data.get('num_ques')
        print(topics)
        print(num_ques)

        context = " ".join(x.page_content for x in docs )
        metadata = docs[0].metadata
      
      
        ans  = generate_response(topics,context,metadata)
       
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
        chain = temp | model | parser 

   
    

        raw = chain.invoke({'content':ans,'num_ques':num_ques})
        raw = clean_llm_json(raw)'''
     

        return Response({"success":True,"content":1})
    except Exception as e:
        return Response({"success": False, "error": str(e)})



