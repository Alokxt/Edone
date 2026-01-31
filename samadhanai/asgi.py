"""
ASGI config for samadhanai project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from chatapp import routing 
from . import settings
from chatapp.middleware import JWTAuthMiddleWare



os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'samadhanai.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": JWTAuthMiddleWare(AuthMiddlewareStack(
        URLRouter(
            routing.websocket_urlpatterns
        ))
    ),
})
