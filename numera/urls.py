from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('visualisation/', views.visualisation, name='visualisation'),
    path('distribution/', views.distribution, name='distribution'),
]
