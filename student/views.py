
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404
import json 
from student.models import * 
from django.core.mail import send_mail
from django.utils.http import urlsafe_base64_encode
from django.utils.http import urlsafe_base64_decode
from django.utils.encoding import force_str
from django.utils.encoding import force_bytes
from django.core.mail import EmailMessage
from django.conf import settings
from rest_framework.views import APIView
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404
from student.data.predictor import prediction
from openai import OpenAI
from django.utils import timezone
from django.contrib import messages
from .models import *
from django.contrib.auth.models import User
import random
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken
from django.http import JsonResponse, Http404
from samadhanai.settings import DEEPSEEK_API_KEY
from student.forms import * 
import random
from django.core.cache import cache
from .serializer import *
import os 

OTP_TTL = 300 
from .tokens import email_verification_token

def get_client():
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=DEEPSEEK_API_KEY,
    )
    return client
def get_groqmodel():
    from langchain_groq import ChatGroq
    model = ChatGroq(
        model_name="openai/gpt-oss-20b",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.7
    )
    return model

@api_view(['POST'])
@permission_classes([AllowAny])
def login_view(request):
    
    try:
        data = request.data 
        username = data.get('username')
        password = data.get('password')
        
        user = authenticate(username = username,password=password)
        
        if user is not None:
            
            refresh = RefreshToken.for_user(user)
            return Response({
                "success": True,
                "message": "Login successful",
                "access": str(refresh.access_token),
                "refresh": str(refresh),
            }, status=status.HTTP_200_OK)
        else:
                return Response({
                "success": False,
                "message": "Invalid username or password"
            }, status=status.HTTP_401_UNAUTHORIZED)
    except Exception as e:
        
        return Response({"success":False,"Message":f"Something went wrong {e}"},status=500)
    


@api_view(["POST"])
@permission_classes([AllowAny])
def send_email_otp(request):
    email = request.data.get("email")

    if not email:
        return Response({"message": "Email required"}, status=400)

    if Myuser.objects.filter(email=email).exists():
        return Response({"message": "Email already registered"}, status=400)

    otp = str(random.randint(100000, 999999))

    cache.set(f"email_otp:{email}", otp, OTP_TTL)
    cache.set(f"email_verified:{email}", False, OTP_TTL)
    try:
        sent = send_mail(
            subject="Your verification code",
            message=f"Your OTP is {otp}. Valid for 5 minutes.",
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[email],
        )

        if sent == 0:
            cache.delete(f"email_otp:{email}")
            return Response({"success":False,"message": "Email failed to send"}, status=500)
        return Response({"success": True, "message": "OTP sent"})
    except Exception as e:
        return Response({"success":False,"message":f"{e}"})

        
    
@api_view(["POST"])
@permission_classes([AllowAny])
def verify_email_otp(request):
    try:

        email = request.data.get("email")
        otp = request.data.get("otp")
        if not email or not otp:
            return Response({"message": "Email and OTP required"}, status=400)

        cached_otp = cache.get(f"email_otp:{email}")

        if not cached_otp:
            return Response({"message": "OTP expired"}, status=400)

        if cached_otp != str(otp):
            return Response({"message": "Invalid OTP"}, status=400)

        cache.set(f"email_verified:{email}", True, OTP_TTL)
        cache.delete(f"email_otp:{email}")

        return Response({"success": True, "message": "Email verified"})
    except Exception as e:
        return Response({"success":False,"Message":f"An error occured , {e}"})


@api_view(['POST'])
@permission_classes([AllowAny])
def singup_view(request):
    
    try:
        data = request.data 
        email = data.get('email')
        username = data.get('username')
        pass1 = data.get('password1')
        pass2 = data.get('password2')

        if not all([email, username, pass1, pass2]):
            return Response({"success": False, "Message": "All fields are required"}, status=400)
        
        if(pass1 != pass2):
            return Response({"Success":False,"Message":"Password Mismatch"},status=400)
        
        if(User.objects.filter(username = username).exists()):
            error = 'user already exist with this username'
            return Response({"success":False,"Message":error},status=400)
        if(Myuser.objects.filter(email = email).exists()):
            error = 'Email already registered'
            return Response({"success":False,"Message":error},status=400)
        
        verified_mail= cache.get(f"email_verified:{email}")
        if not verified_mail :
            return Response({'success':False,"Message":"Email was not verified"})
       
    
        my_user = User.objects.create_user(username=username,password=pass1)
       
        my_user.save()
       
        myuss = Myuser.objects.create(student = my_user,email=email)
        myuss.save()
           
        return Response({"success":True,"Message":"User Registered Successfully"})
    except Exception as e:
        return Response({"Success":False,"Message":"Something went wrong {e}"},status=500)
   

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def dash_board(request):
    try:

        usr = request.user
        myusr = Myuser.objects.filter(student=usr).first()
        profile_url = None
        if myusr and myusr.profile:
            profile_url = request.build_absolute_uri(myusr.profile.url)
        
        details = {
            "Name":usr.username,
            "profile": profile_url 
        }
        return Response({'success':True,'details':details},status=200)
    except Exception as e:
        return Response({'success':False,'error':str(e)},status=500)







@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_roadmap(request):
   
    try:
        road = NewRoadmap.objects.all()
    except roadmap.DoesNotExist:
        return Response({"error": "Roadmap not found"}, status=404)
    registered_roads = NewRoadmap.objects.filter(steps__progress__student = request.user).distinct()
    stps = StudentProgress.objects.filter(student=request.user,completed=True)
    student_roads = [r.name for r in registered_roads]
    done_steps = [s.step.title for s in stps]
    maps = {}
    for r in road:

        info = {}
        steps = r.steps.all()
        if steps:
            for s in steps:
                info[s.title] = {
                    'Description':s.description,
                    'order':s.order,
                    'is_completed': True if s.title in done_steps else False
                } 
        maps[r.name] = {
            'info':info,
            'is_registered':True if r.name in student_roads else False
        }
    
    return Response({'success':True,'roadmap':maps},status=200)    
    
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def mark_complete(request):
    try:
    
        step_name = request.data.get('step_title')
        user_id = request.user.id 
        

        road_name = request.data.get('road_name')
        
        studnt = get_object_or_404(User,id=user_id)
        road = NewRoadmap.objects.filter(name=road_name).first()
        step = RoadmapStep.objects.filter(title=step_name,roadmap=road).first()
        
        
        com = get_object_or_404(StudentProgress,student = studnt,step=step)

        com.completed = True
        com.completed_on = timezone.now()
        com.save()
        return Response({"success":True})
    except Http404:
        return Response({"success":False,"error":'Student Not Registered'},status=404)

def find_progress(road,usr):
    total_steps = road.steps.count()
    all_steps = StudentProgress.objects.filter(
        student=usr,
        step__roadmap=road
    )
    progress = {}
    com =0
    for s in all_steps:
        if s.completed == True:
            progress[s.step.title] = s.completed_on
            com += 1
        else:
            progress[s.step.title] = "Not completed"
    percent = round((com/total_steps)*100,2) if total_steps else 0

    return progress,percent
    



@api_view(['GET'])
@permission_classes([IsAuthenticated])
def show_progress_roadmap(request):
    try:
        road_name = request.GET.get('road_name')
        user_id = request.user.id  
       
        
        stdnt = get_object_or_404(User,id=user_id)
        
        
        road = get_object_or_404(NewRoadmap,name=road_name)
        
        if road is None:
            return Response({'success':False,'message':'No road map with this name'})

        progress,percent = find_progress(road,stdnt)
        
        return Response({"sucess":True,"Progress":progress,'completion_percent':percent})
    except Http404:
        return Response({"success":False,"error":"roadmap not found"},status=404)
        
            

@csrf_exempt
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def predict_performance(request):
    try:
        
        user_id = request.user.id
        if not user_id:
            user_id =2
        stdnt = get_object_or_404(User,id=user_id)
        studies = get_object_or_404(Studyparamms,student=stdnt)
        num_hours = studies.study_hours
        days = studies.days
        av_hrs = num_hours/days 

        road = get_object_or_404(roadmap,student=stdnt)
        subs,per = find_progress(road)
        
        iq = studies.iq_score
        pastper = studies.past_exam_performance
        vals = [av_hrs,per,iq,pastper]
        performance = prediction(vals)
        print(performance)
        return Response({"success":True,"predictive_performance":performance})
    except Exception as e:
        return Response({"success":False,"error":e})




@api_view(['GET'])
@permission_classes([IsAuthenticated])
def subtopics(request):
    try:
        user_id = request.user.id 
        if not user_id:
            user_id = 2 
        stdnt = get_object_or_404(User,id=user_id)
        subs = sub_topics.objects.filter(student=stdnt)
        more_sub = all_sub_topics.objects.all()
        l = []
        for s in subs:
            l.append(s.name)
        for t in more_sub:
            l.append(t.name)
       
        return Response({"success":True,"List_of_subtopics":l})
    except Exception as e:
        return Response({"success":False,"error":e})
    


def generate_test():
    prompt = '''
You are an expert aptitude test paper generator used for competitive exams and placement assessments.

Generate 5 aptitude test questions suitable for students aged 18+.

Rules:
1. Questions must cover quantitative aptitude and logical reasoning, such as:
   - Arithmetic (percentages, ratios, time & work, speed & distance)
   - Number series and patterns
   - Basic probability and averages
   - Logical reasoning and analytical puzzles
2. Each question must be medium to hard difficulty (not trivial calculations).
3. Each question must have exactly 4 options (A, B, C, D).
4. Mark only ONE correct answer per question.
5. Do not repeat question styles.
6. Return the output strictly in the following JSON format (no markdown, no extra text):

{
    "questions": [
        {
            "question": "Question text here",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer": "Correct Option (e.g., B)"
        },
        {
            "question": "...",
            "options": ["...", "...", "...", "..."],
            "answer": "..."
        }
    ]
}
'''
    client = get_client()
    response = client.chat.completions.create(
        model="tngtech/deepseek-r1t2-chimera:free",
        messages=[
            {"role": "system", "content": "You are an expert IQ test designer"},
            {"role":"user","content":prompt},
            
        ],
        temperature=0.3,
    )



    return response.choices[0].message.content

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def register_roadmap(request):
    try:
        user_id =  request.user.id 
        usr = get_object_or_404(User,id = user_id)
        nam = request.data.get('road_name')
        orig = NewRoadmap.objects.filter(name = nam,is_official = True).first()
        
        if orig is None:
            return Response({'success':False,'message':'roadmap not found'},status=400)
        
        for s in orig.steps.all():
            StudentProgress.objects.get_or_create(student=usr,step=s)    
        return Response({'success':True,'message':'registered successfully'},status=200)
    except Http404:
        return Response({'success':False,'error':'Not found'},status=400)



@api_view(['GET'])
@permission_classes([IsAuthenticated])
def iq_test_page(request):
    try:
        user_id = request.user.id 
        
        user = get_object_or_404(User,id=user_id)
        stdnt = get_object_or_404(Myuser,student=user)
        test,created = iq_test.objects.get_or_create(student=stdnt,defaults={
            'date':timezone.now(),
            'clear':False,
            'score':0.0,
        })
       
        if stdnt.needs_quiz() == True:
            
            if created == True:
                
                quize = generate_test()
                test.test_data = quize
                test.save()
                
            elif test.clear == False:
                quize = test.test_data
            else:
                quize = generate_test()
                test.test_data = quize
                test.save()

            return Response({"success":True,"Quize_required":True,"Quiz":quize})
        
        marks = test.score 
        return Response({"success":True,"quiz_required":False,"Marks":1})
    except Exception as e:
        return Response({"success":False,"error":e})
    

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def save_iq_test(request):
    try:
        user_id = request.user.id 
        data = json.loads(request.body)
        score = data.get('score')
        if not user_id:
            user_id = 2
        usr = get_object_or_404(User,id=user_id)
        myusr = get_object_or_404(Myuser,student=usr)
        iq = get_object_or_404(iq_test,student=myusr)
        iq.score = score 
        iq.clear = True 
        myusr.last_iq_quiz = timezone.now()

        iq.date = timezone.now()
        iq.save()
        myusr.save()
        
        return Response({"success":True})
    except Exception as e:
        return Response({"success":False})
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def add_targets(request):
    try:
        user = request.user 
        my_usr = get_object_or_404(Myuser,student=user)
        exams = request.data.get('exams', [])
        if not exams:
            return Response(
                {'success': False, 'message': 'No exams provided'},
                status=400
            )

        for ex in exams:
            exmod = CompExams.objects.filter(name=ex).first()
            if exmod:
                obj = UserPrepration.objects.get_or_create(student=my_usr,exm=exmod)
               
        return Response({'success':True,"Message":"Target Goals are Set"})
    except Exception as e:
        return Response({'success':False,"Message":f"An error occurred {e}"})
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_profile_photo(request):
    try:

        user = request.user
        my_usr = get_object_or_404(Myuser,student=user)

        serializer = ProfilePhotoSerializer(
            my_usr,
            data=request.data,
            partial=True
        )

        if serializer.is_valid():
            serializer.save()
            return Response({
                "success": True,
                "profile_photo": my_usr.profile.url if my_usr.profile else None 
            })

        return Response(serializer.errors, status=400)
    except Exception as e:
        return Response({'success':False,"Message":f"An error occurred {e}"})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def profile_page(request):
    try:
        usr = request.user 
        my_usr = get_object_or_404(Myuser,student=usr)
        perps = UserPrepration.objects.filter(student=my_usr)
        exams = []
        for p in perps:
            exams.append(p.exam.name)
        print(my_usr.profile.url)
        data = {
            "username":usr.username,
            "email":my_usr.email,
            "profile_photo":  request.build_absolute_uri(my_usr.profile.url) if my_usr.profile else None,
            "target_exams":exams
        }

        return Response({'success':True,"Message":"Data found","data":data})
    except Exception as e:
        return Response({'success':False,"Message":f"An error occurred {e}"})
    