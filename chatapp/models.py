from django.db import models
from django.contrib.auth.models import User
from student.models import Myuser 
import shortuuid
from django.core.mail import send_mail
from django.conf import settings


class ChatGroup(models.Model):
    group_name = models.CharField(max_length=250,unique=True,default=shortuuid.uuid)
    users_online = models.ManyToManyField(Myuser,related_name="users_online",blank=True)
    members = models.ManyToManyField(Myuser,related_name="chat_groups",blank=True)
    is_private = models.BooleanField(default=False)

    

    def __str__(self):
        return self.group_name
    

class Groupmessage(models.Model):
    group = models.ForeignKey(ChatGroup,related_name='chat_messages',on_delete=models.CASCADE)
    author= models.ForeignKey(Myuser,on_delete=models.CASCADE)
    body = models.CharField(max_length=300,blank=True,null=True)
    file = models.FileField(upload_to='files/',blank=True,null=True)
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.author.student.username} :{self.body}'
    
    class Meta:
        ordering = ['-created']



class GroupInvite(models.Model):
    group = models.ForeignKey(ChatGroup, on_delete=models.CASCADE, related_name='invites')
    sender = models.ForeignKey(Myuser, on_delete=models.CASCADE, related_name='sent_invites')
    email = models.EmailField()
    token = models.CharField(max_length=64, unique=True, default=shortuuid.uuid)
    accepted = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def send_invite(self):
        invite_link = f"{settings.FRONTEND_URL}/join-group/{self.token}/"
        send_mail(
            subject=f"Invite to join {self.group.group_name}",
            message=f"You've been invited to join {self.group.group_name}. Click to join: {invite_link}",
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[self.email],
        )
