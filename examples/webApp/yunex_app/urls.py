from django.urls import path
from . import views
from yunex_app.views import display_string
from yunex_app.views import receive_json

urlpatterns = [
    path('', views.yunex_app, name='yunex_app'),
    path('get_csrf_token/', views.csrf_token_endpoint, name='get_csrf_token'),
    path('stream/', views.stream_image, name='stream_image'),
    path('display/<str:input_string>/', views.display_string, name='display_string'),
    path('receive_json/', receive_json, name='receive_json'),
    path('run_app/', views.run_external_app, name='run_external_app'),
]