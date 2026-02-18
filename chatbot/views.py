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
import pandas as pd 
from functools import lru_cache
from .embds import get_embeddings
from .getqs import get_matchs 
import numpy as np 
import hashlib
import pickle

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

def get_groqmodel():
    from langchain_groq import ChatGroq
    model = ChatGroq(
        model_name="openai/gpt-oss-20b",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.7
    )
    return model


def get_prompttemp():
    from langchain_core.prompts import PromptTemplate
    return PromptTemplate
def get_strparser():
    from langchain_core.output_parsers import StrOutputParser
    parser = StrOutputParser()
    return parser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

Knowledge_PATH = BASE_DIR / "data2" / "rag_index2"
Question_PATH = BASE_DIR / "data2" / "leetcode_dataset - lc.csv"
"""
EMBEDDING_DIM = 768
INDEX_FILE = BASE_DIR / "data2" / "semantic_cache.index"
MAX_EMBEDDINGS = 100

def get_faiss_index():
    faiss = get_vectorstores()
    if os.path.exists(INDEX_FILE):
        faiss_index = faiss.read_index(INDEX_FILE)
        return faiss_index
    else:
        base_index = faiss.IndexFlatL2(EMBEDDING_DIM)
        faiss_index = faiss.IndexIDMap(base_index)
        return faiss_index


VECTOR_ID_KEY = "semantic:vector_id_counter"
VECTOR_QUEUE_KEY = "semantic:vector_queue"
"""

q_bank = pd.read_csv(Question_PATH)

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=DEEPSEEK_API_KEY,
)
def normalize_query(query: str) -> str:
    return query.strip().lower()

def hash_query(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()


@lru_cache()
def get_vectorr():
   FAISS = get_vectorstores()
   vectorR = FAISS.load_local( str(Knowledge_PATH), get_embeddings(), allow_dangerous_deserialization=True )
   return vectorR

def check_exact_cache(query: str):
    normalized = normalize_query(query)
    key = f"exact:{hash_query(normalized)}"

    response = redis.get(key)
    if response:
        return response

    return None

def store_exact_cache(query: str, response: str):
    normalized = normalize_query(query)
    key = f"exact:{hash_query(normalized)}"

    redis.setex(key, 60 * 60 * 24, response)


"""def check_semantic_cache(query: str):
    faiss_index = get_faiss_index()
    if faiss_index.ntotal == 0:
        return None

    embedding = np.array([get_embeddings(query)]).astype("float32")

    distances, indices = faiss_index.search(embedding, 1)

    best_distance = distances[0][0]
    best_id = indices[0][0]

    if best_id == -1:
        return None

    similarity = 1 / (1 + best_distance)

    if similarity >= 0.90:
        cached_response = redis.get(f"semantic:response:{best_id}")
        if cached_response:
            return cached_response

    return None


def store_semantic_cache(query: str, response: str):
    embedding = np.array([get_embeddings(query)]).astype("float32")

    
    vector_id = redis.incr(VECTOR_ID_KEY)

    faiss_index = get_faiss_index()
    faiss = get_vectorstores()
  
    faiss_index.add_with_ids(
        embedding,
        np.array([vector_id], dtype=np.int64)
    )

    redis.set(f"semantic:response:{vector_id}", response)

 
    redis.rpush(VECTOR_QUEUE_KEY, vector_id)

    total_vectors = faiss_index.ntotal
    if total_vectors > MAX_EMBEDDINGS:
        oldest_id = redis.lpop(VECTOR_QUEUE_KEY)

        if oldest_id:
            oldest_id = int(oldest_id)

            faiss_index.remove_ids(
                np.array([oldest_id], dtype=np.int64)
            )

            redis.delete(f"semantic:response:{oldest_id}")

    faiss.write_index(faiss_index, INDEX_FILE)"""


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
    #model = get_chatmodel()
    model = get_groqmodel()
    parser = get_strparser()
    chain = prompt | model | parser
   
    

    raw = chain.invoke({'context':context,'query':query,'metadata':metadata})

    return raw
    

def get_question(idx):
    PromptTemplate = get_prompttemp()
    pt1 = PromptTemplate(
    template="""
System Prompt: Interviewer Mode (Strict Structured Output)

You are a strict technical interviewer conducting a real Data Structures and Algorithms interview.

Your responsibility is to present the provided question in a professional interview format.


{question}

Behavior Rules:

1.Reformulate the provided question clearly and formally.

2. Do NOT solve the problem.

3.Do NOT reveal the topic name explicitly.

4. You may include 1 or 2 examples ONLY if you are fully certain they are correct.

Each example must include:

1. Input

2. Output

Brief explanation (optional but concise)

1 . You may add a subtle hint if helpful, but:

2. Do NOT reveal the algorithm directly.

3. Do NOT describe the exact technique.

4. Do NOT provide pseudocode or solution steps.

#End by asking the candidate to:

->Explain their approach first.

->Discuss time and space complexity.

Mention edge cases.

Tone Requirements:

Professional

Concise

Neutral

Interview-style

No emojis

No excessive verbosity

Strict Output Format:

You MUST return output in valid JSON format.

Everything (problem statement, examples, constraints, hint, and final instruction to candidate) must be inside a single key named:

{{
"problem": "FULL formatted interview question text here"
}}

Do not return anything outside this JSON.
Do not add extra keys.
Do not include explanations outside the JSON.

The entire formatted interview question must be stored inside the string value of "problem".

""",
input_variables=['question']

)
    question = q_bank.iloc[idx]["description"]
    model = get_groqmodel()
    parser = get_strparser()
    chain = pt1 | model | parser 
    ans = chain.invoke({"question":question})
    return ans 

def get_evaluation(idx,resp):
    question = q_bank.iloc[idx]["description"]
    PromptTemplate = get_prompttemp()
    pt2 = PromptTemplate(
        template="""
You are a strict technical interviewer evaluating a candidate in a Data Structures and Algorithms interview.

You will be given:

The original interview question : {question}

The candidate’s response (approach and explanation): {resp}

Your responsibilities:

Carefully evaluate the candidate’s response.

Determine whether the core idea is correct.

Assess:

Logical correctness

Efficiency awareness

Handling of edge cases

Clarity of explanation

If the solution is partially correct, clearly explain what is missing.

If the approach is incorrect, explain why.

Be objective and precise.

Do NOT rewrite the full solution.

Do NOT reveal the exact optimal solution unless absolutely necessary for clarity.

Generate one relevant follow-up question:

It must relate to the same problem.

It should increase depth (optimization, edge case, constraint change, or alternative approach).

It must not reveal the topic name explicitly.

Scoring Rules:

Score must be an integer from 0 to 10.

9–10: Strong and near-optimal.

6–8: Mostly correct but missing depth or edge cases.

3–5: Partially correct with significant gaps.

0–2: Incorrect or fundamentally flawed.

Strict Output Format:

Return ONLY valid JSON in this exact structure:

{{
"insights": {{
"explaination": "concise evaluation of the candidate’s response. Should not be more than 2-3 sentences.",
"score": "integer between 0 and 10"
}},
"followup": "A single, clear follow-up interview question."
}}

Do not include anything outside this JSON.
Do not add extra keys.
Ensure the JSON is valid.
""",
input_variables=["question","resp"]
)
    model = get_groqmodel()
    parser = get_strparser()
    chain = pt2 | model | parser 
    rep = chain.invoke({"question":question,"resp":resp})
    return rep 

"""
@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def home(request):
    try:
        
        query = request.data.get('query')
       
      
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
"""

@csrf_exempt
@api_view(["POST"])
@permission_classes([AllowAny])
def chatbot_api(request):
    try:
        query = request.data.get("query")

        if not query:
            return Response({"success": False, "error": "Query required"})

        
        exact_response = check_exact_cache(query)
        if exact_response:
            return Response({"success": True, "answer": exact_response, "cache": "exact"})


        vectorR = get_vectorr()
        docs = vectorR.similarity_search(
            query,
            k=2
        )

        
        context = " ".join(x.page_content for x in docs )
        metadata = docs[0].metadata
        llm_response = generate_response(query,context,metadata)

        
        store_exact_cache(query, llm_response)
        #store_semantic_cache(query, llm_response)

        return Response({"success": True, "answer": llm_response, "cache": "new"})
    except Exception as e:
        print(e)
        return Response({"success":False,"error":f"{e}"},status=500)

    
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
    #model = get_chatmodel()
    model = get_groqmodel()
    parser = get_strparser()
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
    
 

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def youtubevideo(request):
    try:
        data = request.data 
        video_url = data.get('video_url','')
        lang = data.get('language')
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
       data = request.data 
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
  
@csrf_exempt
@api_view(["POST"])
@permission_classes([AllowAny])
def quiz_generator(request):
    try:

        data = request.data 
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
        #model = get_chatmodel()
        model = get_groqmodel()
        parser = get_strparser()
        chain = temp | model | parser

   
    

        raw = chain.invoke({'content':ans,'num_ques':num_ques})
        raw = clean_llm_json(raw)
     

        return Response({"success":True,"content":raw})
    except Exception as e:
        return Response({"success": False, "error": str(e)},status=500)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def Interview_simulator(request):
    try:
        data = request.data 
        user = request.user 
        key = f"interviewkey_{user.id}"
        insight_key = f"Insights_{user.id}"
        old = redis.hgetall(key)
        
        if old:
            pos = int(redis.hget(key,"current_pos"))
            if pos > 3:
                scores = int(redis.hget(key,"scores"))
                insights = [json.loads(i) for i in redis.lrange(insight_key, 0, -1)]
                if scores is None:
                    scores = 25
                content = {
                    "interview_completed":True,
                    "score":scores,
                    "insights":insights
                }
                redis.delete(key)
                redis.delete(insight_key)
                
                return Response({"success":True,"data":content})
            
            q_key = f"q{pos}_index"
            q_state = f"q{pos}_state"
           
            stat = old[q_state]
            idx = int(old[q_key])
            
            if stat == "start":
                content = get_question(idx)
                content = json.loads(content)
                redis.hset(key,q_state,"attempted")
                return Response({'success':True,"data":content})
                
            elif stat == "attempted":
                if "answer" not in data:
                    redis.delete(key)
                    redis.delete(insight_key)
                    return Response({'sucess':False,"message":"No response for Q1"})
                resp = data["answer"]
                content = get_evaluation(idx,resp)
                content = json.loads(content)
                sc  = int(content["insights"]["score"])
                if sc is None:
                    sc =5

                insights = content["insights"]["explaination"]
                if insights is None:
                    insights = ""
                insight_dict = {
                    "question":pos,
                    "score":sc,
                    "Evaluation":insights
                }
                redis.rpush(insight_key,json.dumps(insight_dict))

                p_sc = int(redis.hget(key,"scores"))
                if p_sc is None:
                    p_sc = 0 
                scr = str(sc+p_sc)
             
               
                redis.hset(key,"scores",scr)
              
                redis.hset(key,q_state,"passed")
                d = {
                    "interview_completed":False,
                    "problem":content["followup"]
                }

              
                redis.hset(key,"current_pos",pos+1)
                return Response({'success':True,"data":d})

            else:
                return Response({'success':False,"message":"something broke"})
        else:
            ques = get_matchs(3)
            states = {
                "current_pos": 1,
                "q1_index": ques[0],
                "q1_state": "start",
                "q2_index": ques[1],
                "q2_state": "start",
                "q3_index": ques[2],
                "q3_state": "start",
                "scores": 0,
            }
            
            redis.hmset(key,states)
            redis.expire(insight_key, 3600)
            redis.expire(key,3600)


            return Response({'success':True,"Message":"Session started"})
    except Exception as e:
        print(e)
        return Response({'success':False,"error":f"{e}"},status=500)
        

                
