from django.contrib import admin
from django.urls import path,include
from chatbot import views 
urlpatterns = [
    path('',views.chatbot_api , name="homepage"),
    path('chat-video/',views.chatvideo,name="chat_video"),
    path('process-video/',views.youtubevideo,name="process_video"),
    path('generate-quiz/',views.quiz_generator,name="quiz_generator"),
    path('interview/',views.Interview_simulator,name="interview_simulator"),
    
]