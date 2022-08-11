from django import forms
from .models import Scoring


class scoreform(forms.ModelForm):
    class Meta:
        model = Scoring
        fields = "__all__"
