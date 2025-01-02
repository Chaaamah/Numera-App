from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('home/', views.home, name='home'),
    path('visualisation/', views.visualisation, name='visualisation'),
    path('visualisation2/', views.visualisation2, name='visualisation2'),
    path('distribution/', views.distribution, name='distribution'),
    path('distribution2/', views.distribution2, name='distribution2'),
    path('hypothesis/', views.hypothesis, name='hypothesis'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('profile/', views.profile_view, name='profile'),
    path('logout/', views.logout_view, name='logout'),
]