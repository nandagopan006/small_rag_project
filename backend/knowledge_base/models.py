
from django.db import models


class PolicyDocument(models.Model):
    class Status(models.TextChoices):
        UPLOADED = "uploaded", "Uploaded"
        PROCESSING = "processing", "Processing"
        PROCESSED = "processed", "Processed"
        FAILED = "failed", "Failed"

    title = models.CharField(max_length=200)
    file = models.FileField(upload_to="policies/")
    version = models.PositiveIntegerField(default=1)

    # PDF processing metadata
    page_count = models.PositiveIntegerField(default=0)
    extracted_pages = models.JSONField(default=list, blank=True)

    # Processing status
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.UPLOADED,
    )

    uploaded_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title
