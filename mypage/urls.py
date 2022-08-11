from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('newsScoring', views.newsScoring, name='newsScoring'),
    path('label', views.label, name='label'),
    path('edit/<int:id>', views.edit, name="edit"),
    path('update/<int:id>', views.update, name="update"),
    path('classify', views.classify, name='classify'),
    path('reset', views.reset, name='reset'),
    path('new', views.newS, name='new'),
]
