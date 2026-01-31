from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from student.models import *

class RegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ['username','password1','password2','email']

'''class GroupChatMessage(forms.ModelForm):
    class Meta:
        model = Groupmessage
        fields =['body']
        widgets={
            'body':forms.TextInput(attrs={'placeholder':'Add message...','class':'p-4 text-black','maxlength':'300','autofocus':True}),
        }


class ChatGroupForm(forms.Form):
    Group_name = forms.CharField(max_length=250)
    members = forms.ModelMultipleChoiceField(queryset = Myuser.objects.all(),widget=forms.CheckboxSelectMultiple)
    

class SelectCountry(forms.Form):
    Country = forms.ModelMultipleChoiceField(queryset=country.objects.all(),widget=forms.Select)
    '''