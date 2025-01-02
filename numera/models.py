

# Create your models here.
from django.db import models
from django.contrib.auth.models import User

class UserActivity(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    activity_type = models.CharField(max_length=50)  # 'visualization', 'distribution', 'data_import'
    timestamp = models.DateTimeField(auto_now_add=True)
    description = models.TextField()

class DatasetUpload(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    filename = models.CharField(max_length=255)
    upload_date = models.DateTimeField(auto_now_add=True)

class Analysis(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    analysis_type = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)
    results = models.JSONField()