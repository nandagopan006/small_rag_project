from rest_framework import serializers
from .models import PolicyDocument

from pathlib import Path

class PolicyDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = PolicyDocument
        fields = ["id",
            "title",
            "file",
            "version",
            "status",
            "uploaded_at",
            "updated_at",]
        
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

        if value.size > 20 * 1024 * 1024:
            raise serializers.ValidationError(
                "The maximum file size is 20 MB."
            )

        return value
        
        



