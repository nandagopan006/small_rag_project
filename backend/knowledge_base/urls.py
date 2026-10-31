
from django.urls import path

from .views import PolicyDocumentAPIView

urlpatterns = [
    path(
        "documents/",
        PolicyDocumentAPIView.as_view(),
        name="document-list-create",
    ),
]