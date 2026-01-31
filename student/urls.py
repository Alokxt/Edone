from django.contrib import admin
from django.urls import path,include
from rest_framework_simplejwt.views import TokenRefreshView
from student.views import*

urlpatterns = [
    path('roadmaps/',get_roadmap, name="roadmaps"),
    path('markcompleted/',mark_complete,name="iscompleted"),
    path('showprogress/',show_progress_roadmap,name="progress"),
    path('predictperformance/',predict_performance,name="performancepredictions"),
    path('subtopics/',subtopics,name='gettopics'),
    path('iqquiz/',iq_test_page,name='iqquize'),
    path('savequiz/',save_iq_test,name='saveiqquiz'),
    path('signup/',singup_view,name='singup'),
    path('login/',login_view,name='login_page'),
    path('dashboard/',dash_board,name='dashboard'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('register/roadmap/',register_roadmap,name='register'),
    path('send-otp/',send_email_otp,name='sendotp'),
    path("verify-email/",verify_email_otp,name='varifymail'),
    path('profile/upload-photo/', upload_profile_photo,name="upload_profile"),
    path('profile/',profile_page,name="user_profile"),
]