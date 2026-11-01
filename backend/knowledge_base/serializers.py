from pathlib import Path
from rest_framework import serializers

from .models import PolicyDocument


class PolicyDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = PolicyDocument
        fields = [
            "id",
            "title",
            "file",
            "version",
            "status",
            "uploaded_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "status",
            "uploaded_at",
            "updated_at",
        ]

    def validate_file(self, value):
        if Path(value.name).suffix.lower() != ".pdf":
            raise serializers.ValidationError(
                "Only PDF files are allowed."
            )

        if value.size == 0:
            raise serializers.ValidationError(
                "The uploaded file is empty."
            )

        if value.size > 20 * 1024 * 1024:
            raise serializers.ValidationError(
                "The maximum file size is 20 MB."
            )

        return value


class AskQuestionSerializer(serializers.Serializer):
    """Validate data for asking questions about college policies."""

    question = serializers.CharField(
        max_length=2000,
        trim_whitespace=True,
    )
    top_k = serializers.IntegerField(
        required=False,
        default=5,
        min_value=1,
        max_value=10,
    )
    document_id = serializers.IntegerField(
        required=False,
        allow_null=True,
        min_value=1,
    )
    min_similarity = serializers.FloatField(
        required=False,
        default=None,
        allow_null=True,
        min_value=0.0,
        max_value=1.0,
    )

    def validate_document_id(self, value):
        """Check whether the requested document exists and is processed."""
        if value is None:
            return value

        document = PolicyDocument.objects.filter(id=value).first()

        if document is None:
            raise serializers.ValidationError(
                "The requested document does not exist."
            )

        if document.status != PolicyDocument.Status.PROCESSED:
            raise serializers.ValidationError(
                "The requested document is not yet processed."
            )

        return value
