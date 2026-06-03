from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_document, name='upload_document'),
    path('query/', views.query_page, name='query_page'),
    path('ask/', views.ask_question, name='ask_question'),
    path('evaluation/', views.evaluation_page, name='evaluation_page'),
    path('run-evaluation/', views.run_evaluation, name='run_evaluation'),
]