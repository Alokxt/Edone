from django.contrib import admin
from django.urls import path,include
from chatbot.views import*
urlpatterns = [
    path('',home , name="homepage"),
    path('chat-video/',chatvideo,name="chat_video"),
    path('process-video/',youtubevideo,name="process_video"),
    path('generate-quiz/',quiz_generator,name="quiz_generator"),
]