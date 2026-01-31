import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from chatapp.models import ChatGroup, Groupmessage
from student.models import Myuser

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        print('connect hitted')
        user = self.scope['user']
        self.user = await self.get_user(user)
        
        self.chatroom_name = self.scope['url_route']['kwargs']['Chatroom_name']
        self.chatroom = await self.get_chatroom(self.chatroom_name)

        await self.channel_layer.group_add(self.chatroom_name, self.channel_name)
        user_online = await self.check_if_online(user)

        if user_online == False:
            await self.add_online_user(self.user)
            await self.update_online_count()

        await self.accept()

    async def disconnect(self, close_code):
        user_online = await self.check_if_online(self.user)
        if user_online == True:
            await self.remove_online_user(self.user)
            await self.update_online_count()
        await self.channel_layer.group_discard(self.chatroom_name, self.channel_name)

    async def receive(self, text_data):
        
        data = json.loads(text_data)
        body = data.get("body")

        msg = await self.create_message(self.chatroom, self.user, body)
              
        usr_name = await self.get_username()
        await self.channel_layer.group_send(
            self.chatroom_name,
            {
                "type": "chat_message",
                "message": {
                    "id": msg.id,
                    "author": usr_name,
                    "body": msg.body,
                    "created": msg.created.isoformat(),
                },
            },
        )

    async def chat_message(self, event):
       
        await self.send(text_data=json.dumps({
            "type": "message",
            "message": event["message"]
        }))

    async def update_online_count(self):
        count = await self.get_online_count()
        await self.channel_layer.group_send(
            self.chatroom_name,
            {
                "type": "online_count_update",
                "count": count,
            },
        )

    async def online_count_update(self, event):
        await self.send(text_data=json.dumps({
            "type": "online_count",
            "count": event["count"],
        }))

    
    @database_sync_to_async
    def get_chatroom(self, name):
        return ChatGroup.objects.get(group_name=name)

    @database_sync_to_async
    def create_message(self, chatroom, user, body):
        return Groupmessage.objects.create(group=chatroom, author=user, body=body)
    
    @database_sync_to_async
    def get_username(self):
        return self.user.student.username 
    @database_sync_to_async
    def get_user(self,user):
        return Myuser.objects.filter(student=user).first()

    @database_sync_to_async
    def add_online_user(self, user):
        self.chatroom.users_online.add(user)

    @database_sync_to_async
    def check_if_online(self,user):
        if self.user in self.chatroom.users_online.all():
            return True
        return False

    @database_sync_to_async
    def remove_online_user(self, user):
        self.chatroom.users_online.remove(user)

    @database_sync_to_async
    def get_online_count(self):
        return self.chatroom.users_online.count()
