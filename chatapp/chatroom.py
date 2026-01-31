from channels.generic.websocket import AsyncWebsocketConsumer
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

class ChatRoomConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def recieve(self, text_data):
        await self.send(text_data=text_data)

    async def chat_message(self,event):
        await self.send(text_data=event['message'])