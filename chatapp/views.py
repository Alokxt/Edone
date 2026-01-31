from django.shortcuts import render,redirect
from django.shortcuts import get_object_or_404
from.models import *
from django.http import Http404
from rest_framework.response import Response
from django.contrib.auth.decorators import login_required
from student.models import *
from chatapp.models import *
from rest_framework.permissions import IsAuthenticated

from rest_framework.views import APIView
from chatapp.serializers import *


import shortuuid
from chatapp.models import GroupInvite
from django.conf import settings
from django.core.mail import send_mail



class GroupList(APIView):
    permission_classes = [IsAuthenticated]

    def get(self,request):
        
        my_usr = Myuser.objects.get(student=request.user) 

        public_groups = ChatGroup.objects.filter(is_private = False)
        private_groups = ChatGroup.objects.filter(is_private=True,members=my_usr)
        groups = public_groups.union(private_groups)
        serializer = ChatGroupSerializer(groups, many=True)
        return Response(serializer.data)
    

class GroupMessages(APIView):
    permission_classes = [IsAuthenticated]

    def get(self,request,group_name):
        group = get_object_or_404(ChatGroup, group_name=group_name)
        messages = group.chat_messages.all().order_by('-created')[:50]
        serializer = GroupMessageSerializer(messages, many=True,context={'request':request})
       
        return Response(serializer.data)
    
class CreateGroupAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        name = request.data.get('group_name')
        is_private = request.data.get('is_private', False)
        if ChatGroup.objects.filter(group_name=name).exists():
            return Response({'error': 'Group name already exists'}, status=400)

        group = ChatGroup.objects.create(group_name=name, is_private=is_private)
        group.members.add(request.user)
        return Response({'success': True, 'group_name': group.group_name})



class SendGroupInviteAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        group_name = request.data.get('group_name')
        email = request.data.get('email')

        try:
            group = ChatGroup.objects.get(group_name=group_name)
        except ChatGroup.DoesNotExist:
            return Response({'error': 'Group not found'}, status=404)

        if not group.is_private:
            return Response({'error': 'Invites only allowed for private groups'}, status=400)

        token = shortuuid.uuid()
        invite = GroupInvite.objects.create(group=group, sender=request.user, email=email, token=token)
        invite_link = f"{settings.FRONTEND_URL}/join-group/{token}/"

        send_mail(
            subject=f"Invite to join {group.group_name}",
            message=f"You've been invited to join {group.group_name}. Click here to join: {invite_link}",
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[email],
        )

        return Response({'success': True, 'message': 'Invite sent successfully'})




class AcceptInviteAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, token):
        try:
            invite = GroupInvite.objects.get(token=token, accepted=False)
        except GroupInvite.DoesNotExist:
            return Response({'error': 'Invalid or expired invite'}, status=400)

        invite.group.members.add(request.user)
        invite.accepted = True
        invite.save()

        return Response({'success': True, 'group_name': invite.group.group_name})



@login_required
def chatpage(request,Chatroom_name = "public-chat"):
    try:
        chatgrp = get_object_or_404(ChatGroup,group_name=Chatroom_name)
        chtmsg = chatgrp.chat_messages.all()[:80]
        my_usr = get_object_or_404(Myuser,student=request.user)

        if request.method == "POST":
            body = request.data.get('body')
            grp_msg = Groupmessage.objects.create(
                body = body,
                author = my_usr,
                group = chatgrp,
                created = timezone.now()
            )
            
            
            
            grp_msg.save()
        
            context ={
                'message':grp_msg,
                'user':request.user
            }
            return Response({'success':True,'context':context},status=200)
        context = {
            'chat_messages':chtmsg,
            'chatroom_name':Chatroom_name,
        }
        return Response({'success':True,'context':context})
    except Http404:
        return Response({'success':False,'error':'Not found'},status=400)
    
    

