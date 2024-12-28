from django import forms

class FileUploadForm(forms.Form):
    file = forms.FileField(label="Charger un fichier Excel...")

    def clean_file(self):
        file = self.cleaned_data.get('file')
        if not file.name.endswith(('.csv', '.xlsx', '.xls')):
            raise forms.ValidationError("Seuls les fichiers Excel ou CSV sont acceptés.")
        return file
