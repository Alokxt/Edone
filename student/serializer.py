from rest_framework import serializers
from .models import* 
class ProfilePhotoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Myuser
        fields = ['profile']
