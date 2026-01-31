from urllib.parse import parse_qs
from rest_framework_simplejwt.tokens import AccessToken
from channels.middleware import BaseMiddleware
from django.contrib.auth.models import AnonymousUser
from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model

User = get_user_model()

@database_sync_to_async
def get_user(validated_token):
    try:
        user_id = validated_token['user_id']
        return User.objects.get(id=user_id)
    except User.DoesNotExist:
        return AnonymousUser()

class JWTAuthMiddleWare(BaseMiddleware):
    async def __call__(self, scope, receive, send):
        query = scope['query_string'].decode()
        parsed = parse_qs(query)
       
        token = parsed.get('token', [None])[0]

        if token:
            try:
                access_token = AccessToken(token)
                user = await get_user(access_token)
                scope['user'] = user
            except Exception as e:
                scope['user'] = AnonymousUser()
        else:
            scope['user'] = AnonymousUser()

        return await super().__call__(scope, receive, send)



    