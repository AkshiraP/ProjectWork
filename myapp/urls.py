from django.urls import path
from .views import home, classify_comment

app_name = 'myapp'

urlpatterns = [
    path('', home, name='home'),
    path('classify/', classify_comment, name='classify_comment'),
]