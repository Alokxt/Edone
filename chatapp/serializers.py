from rest_framework import serializers
from chatapp.models import ChatGroup, Groupmessage

class ChatGroupSerializer(serializers.ModelSerializer):
    members_count = serializers.IntegerField(source='members.count', read_only=True)
    
    class Meta:
        model = ChatGroup
        fields = ['id', 'group_name', 'is_private', 'members_count']

class GroupMessageSerializer(serializers.ModelSerializer):
    author = serializers.CharField(source='author.student.username', read_only=True)
    is_user = serializers.SerializerMethodField() 
    class Meta:
        model = Groupmessage
        fields = ['id', 'author', 'body', 'created','is_user']

    def get_is_user(self, obj):
        request = self.context.get('request')
        if request is None:
            return False
        return obj.author.student == request.user