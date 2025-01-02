from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-input',
            'placeholder': 'Adresse email'
        })
    )

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")
        
class FileUploadForm(forms.Form):
    file = forms.FileField(label="Charger un fichier Excel...")

    def clean_file(self):
        file = self.cleaned_data.get('file')
        if not file.name.endswith(('.csv', '.xlsx', '.xls')):
            raise forms.ValidationError("Seuls les fichiers Excel ou CSV sont acceptés.")
        return file
    
