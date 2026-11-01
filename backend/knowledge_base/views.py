import logging

from django.db import transaction
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import DocumentChunk, PolicyDocument
from .serializers import PolicyDocumentSerializer,AskQuestionSerializer
from .services.chunking import chunk_extracted_pages
from .services.pdf_loader import PDFProcessingError, extract_pdf_pages
from .services.vector_store import sync_document_chunks
from .services.rag import ask_policy_question
from .services.llm import LLMServiceError

logger = logging.getLogger(__name__)


class PolicyDocumentAPIView(APIView):

    parser_classes = (MultiPartParser, FormParser)

    def get(self, request):
        documents = PolicyDocument.objects.all().order_by("-uploaded_at")

        serializer = PolicyDocumentSerializer(
            documents,
            many=True,
            context={"request": request}
        )

        return Response(
            serializer.data,
            status=status.HTTP_200_OK
        )

    def post(self, request):
        serializer = PolicyDocumentSerializer(
            data=request.data,
            context={"request": request}
        )

        if serializer.is_valid():
            serializer.save()

            return Response(
                serializer.data,
                status=status.HTTP_201_CREATED
            )

        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )


class PolicyDocumentProcessAPIView(APIView):
    """Extract, chunk, embed, and persist text from an uploaded policy PDF."""

    def post(self, request, pk):
        document = get_object_or_404(
            PolicyDocument,
            pk=pk,
        )

        if document.status == PolicyDocument.Status.PROCESSING:
            return Response(
                {"detail": "Document is already being processed."},
                status=status.HTTP_409_CONFLICT,
            )

        document.status = PolicyDocument.Status.PROCESSING
        document.save(update_fields=["status", "updated_at"])

        try:
            #pdf
            pages = extract_pdf_pages(document.file)
            #split to chunk akkum
            chunks = chunk_extracted_pages(pages)

            if not chunks:
                raise PDFProcessingError(
                    "No extractable chunks were generated from the PDF."
                )

            chunk_objects = [
                DocumentChunk(
                    document=document,
                    chunk_index=chunk["chunk_index"],
                    page_number=chunk["page"],
                    section_title=chunk["section_title"],
                    content=chunk["text"],
                )
                for chunk in chunks
            ]

            with transaction.atomic():
                document.chunks.all().delete()
                DocumentChunk.objects.bulk_create(chunk_objects)

                document.extracted_pages = pages
                document.page_count = len(pages)
                document.save(
                    update_fields=[
                        "extracted_pages",
                        "page_count",
                        "updated_at",
                    ]
                )

        except PDFProcessingError as exc:
            document.status = PolicyDocument.Status.FAILED
            document.save(update_fields=["status", "updated_at"])

            return Response(
                {"detail": str(exc)},
                status=status.HTTP_400_BAD_REQUEST,
            )

        except Exception:
            logger.exception(
                "Unexpected error during PDF extraction or chunk saving for document %s",
                document.pk,
            )
            document.status = PolicyDocument.Status.FAILED
            document.save(update_fields=["status", "updated_at"])

            return Response(
                {"detail": "An unexpected error occurred while processing the PDF."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        try:
            indexed_count = sync_document_chunks(document.pk)
        except Exception:
            logger.exception(
                "Failed to synchronize vector embeddings in ChromaDB for document %s",
                document.pk,
            )
            document.status = PolicyDocument.Status.FAILED
            document.save(update_fields=["status", "updated_at"])

            return Response(
                {"detail": "An unexpected error occurred while indexing document vectors."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        document.status = PolicyDocument.Status.PROCESSED
        document.save(update_fields=["status", "updated_at"])

        return Response(
            {
                "id": document.pk,
                "title": document.title,
                "status": document.status,
                "page_count": document.page_count,
                "chunk_count": len(chunk_objects),
                "indexed_count": indexed_count,
                "message": "PDF processed and indexed successfully.",
            },
            status=status.HTTP_200_OK,
        )


# Alias for backward compatibility
PolicyDocumentProcessApiView = PolicyDocumentProcessAPIView


class AskQuestionAPIView(APIView):
    """Handle user questions about college policies."""

    def post(self, request):
        serializer = AskQuestionSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST,
            )

        question_data = serializer.validated_data

        try:
            result = ask_policy_question(
                question=question_data["question"],
                top_k=question_data.get("top_k", 5),
                document_id=question_data.get("document_id"),
                min_similarity=question_data.get("min_similarity"),
            )
        except ValueError as error:
            return Response(
                {"detail": str(error), "error": str(error)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except LLMServiceError:
            return Response(
                {
                    "detail": "The AI service is currently unavailable. Please try again later.",
                    "error": "The AI service is currently unavailable. Please try again later.",
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception:
            logger.exception("Unexpected error while answering policy question.")
            return Response(
                {
                    "detail": "An unexpected error occurred.",
                    "error": "An unexpected error occurred.",
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            result,
            status=status.HTTP_200_OK,
        )