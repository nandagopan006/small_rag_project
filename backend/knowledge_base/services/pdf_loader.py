

import logging

import fitz

logger = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be processed safely."""


def extract_pdf_pages(file_field):
    """
    Extract text from each page of a Django FileField.

    Returns:
        list[dict]: Page numbers and extracted text.

    Raises:
        PDFProcessingError: If the PDF is invalid, encrypted,
            empty, or contains no extractable text.
    """
    try:
        with file_field.open("rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        if not pdf_bytes:
            raise PDFProcessingError("The uploaded PDF is empty.")

        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.is_encrypted:
                raise PDFProcessingError(
                    "Password-protected PDFs are not supported."
                )

            if document.page_count == 0:
                raise PDFProcessingError(
                    "The PDF does not contain any pages."
                )

            extracted_pages = []

            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text("text").strip()

                extracted_pages.append(
                    {
                        "page": page_number,
                        "text": page_text,
                    }
                )

            if not any(page["text"] for page in extracted_pages):
                raise PDFProcessingError(
                    "No extractable text was found in the PDF."
                )

            return extracted_pages

    except PDFProcessingError:
        raise

    except fitz.FileDataError as exc:
        logger.warning("Unable to read uploaded PDF: %s", exc)
        raise PDFProcessingError(
            "The uploaded file is not a valid, readable PDF."
        ) from exc

    except (fitz.EmptyFileError, ValueError) as exc:
        logger.warning("Unable to process uploaded PDF: %s", exc)
        raise PDFProcessingError(
            "The uploaded file is empty or invalid."
        ) from exc

    except OSError as exc:
        logger.exception("Unable to access the uploaded PDF.")
        raise PDFProcessingError(
            "The uploaded PDF could not be accessed."
        ) from exc
