from django.contrib import admin
from django.urls import path
from chatapp.views import *

urlpatterns = [
    path('groups/', GroupList.as_view()),
    path('groups/create/', CreateGroupAPIView.as_view()),
    path('<str:group_name>/messages/', GroupMessages.as_view()),
    path('invite/', SendGroupInviteAPIView.as_view()),
    path('accept/<str:token>/', AcceptInviteAPIView.as_view()),
    
]