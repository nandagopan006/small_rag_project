
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


class DocumentChunk(models.Model):
    """A text chunk extracted from a policy document."""

    document = models.ForeignKey(
        PolicyDocument,
        on_delete=models.CASCADE,
        related_name="chunks",
    )

    chunk_index = models.PositiveIntegerField()
    page_number = models.PositiveIntegerField()
    section_title = models.CharField(
        max_length=200,
        default="General",
    )
    content = models.TextField()

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["document_id", "chunk_index"]
        constraints = [
            models.UniqueConstraint(
                fields=["document", "chunk_index"],
                name="unique_document_chunk_index",
            ),
        ]
        indexes = [
            models.Index(
                fields=["document", "page_number"],
                name="chunk_document_page_idx",
            ),
        ]

    def __str__(self):
        return f"{self.document.title} - Chunk {self.chunk_index}"
