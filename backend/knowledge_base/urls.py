from django.urls import path

from .views import PolicyDocumentAPIView, PolicyDocumentProcessAPIView

urlpatterns = [
    path(
        "documents/",
        PolicyDocumentAPIView.as_view(),
        name="document-list-create",
    ),
    path(
        "documents/<int:pk>/process/",
        PolicyDocumentProcessAPIView.as_view(),
        name="document-process",
    ),
]
