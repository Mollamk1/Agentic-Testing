"""
Flask REST API for document text extraction.
Provides a /api/upload endpoint that accepts PDF, DOCX, and XLSX files,
extracts text using document_reader.py, and returns metadata + extracted text.
"""

import logging
import os
import time
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from document_reader import extract_text_from_file
from models import DocumentData
from extraction_service import evaluate_compliance

app = Flask(__name__, static_folder=".", static_url_path="")

# Restrict CORS origins: allow all in dev, specific origins in production.
# Set ALLOWED_ORIGINS env var to a comma-separated list for production.
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
_cors_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()] or "*"
CORS(app, origins=_cors_origins)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {"pdf", "docx", "xlsx"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/msword",
    "application/vnd.ms-excel",
}


def _allowed_file(filename: str) -> bool:
    """Return True if the file extension is in the allowed set."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _human_size(num_bytes: int) -> str:
    """Convert a byte count to a human-readable string (KB / MB)."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.route("/")
def index():
    """Serve the frontend UI."""
    return send_from_directory(".", "index.html")


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """
    Accept a file upload, extract text, and return structured results.

    Request:  multipart/form-data with field 'file'
    Response: JSON with keys: success, text, metadata (on success)
                          or: success, error            (on failure)
    """
    # ---- 1. Check that a file was sent ------------------------------------ #
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided."}), 400

    file = request.files["file"]

    if file.filename == "" or file.filename is None:
        return jsonify({"success": False, "error": "No file selected."}), 400

    # ---- 2. Validate file extension --------------------------------------- #
    if not _allowed_file(file.filename):
        return jsonify(
            {
                "success": False,
                "error": (
                    f"Unsupported file type. "
                    f"Please upload a PDF, DOCX, or XLSX file."
                ),
            }
        ), 415

    # ---- 3. Read file bytes and check size -------------------------------- #
    file_bytes = file.read()
    file_size = len(file_bytes)

    if file_size > MAX_FILE_SIZE_BYTES:
        return jsonify(
            {
                "success": False,
                "error": (
                    f"File is too large ({_human_size(file_size)}). "
                    f"Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
                ),
            }
        ), 413

    if file_size == 0:
        return jsonify({"success": False, "error": "Uploaded file is empty."}), 400

    # ---- 4. Save to a temp file and extract text -------------------------- #
    # Use a validated, hardcoded extension from ALLOWED_EXTENSIONS to avoid
    # any path-injection risk from user-controlled filenames.
    safe_extension = file.filename.rsplit(".", 1)[1].lower()
    # safe_extension is guaranteed to be in ALLOWED_EXTENSIONS at this point
    # (checked by _allowed_file above), so we use it directly as the suffix.
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix="." + safe_extension, delete=False
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        start_time = time.time()
        extracted_text = extract_text_from_file(tmp_path)
        elapsed = round(time.time() - start_time, 2)

        # ---- 5. Build metadata ------------------------------------------- #
        char_count = len(extracted_text)
        # Count pages by counting "--- Page" markers inserted by extract_text_from_pdf
        page_markers = extracted_text.count("--- Page ")
        pages = page_markers if page_markers > 0 else None

        metadata = {
            "file_name": file.filename,
            "file_size": _human_size(file_size),
            "file_size_bytes": file_size,
            "char_count": char_count,
            "word_count": len(extracted_text.split()),
            "extraction_time_seconds": elapsed,
        }
        if pages is not None:
            metadata["pages"] = pages

        return jsonify(
            {
                "success": True,
                "text": extracted_text,
                "metadata": metadata,
            }
        )

    except (FileNotFoundError, ValueError) as exc:
        # These are known, safe errors from document_reader – safe to surface.
        logger.warning("Validation error during extraction: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 400

    except Exception as exc:
        # Log the full exception server-side; return a generic message to the
        # client to avoid leaking internal stack-trace details.
        logger.exception("Unexpected error during extraction")
        return jsonify(
            {
                "success": False,
                "error": "An internal error occurred while extracting the file. "
                         "Please try again or contact support.",
            }
        ), 500

    finally:
        # Always clean up the temporary file
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/api/health", methods=["GET"])
def health():
    """Simple health-check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """
    Accept extracted DocumentData as JSON, run compliance evaluation, and return results.

    Request:  application/json matching the DocumentData schema
    Response: JSON with keys: success, data (DocumentData fields), compliance (status + reasons)
              or: success, error (on failure)
    """
    if not request.is_json:
        return jsonify({"success": False, "error": "Request must be JSON."}), 415

    try:
        doc_data = DocumentData(**request.get_json())
    except Exception:
        logger.warning("Invalid DocumentData payload", exc_info=True)
        return jsonify({"success": False, "error": "Invalid payload: could not parse DocumentData fields."}), 400

    compliance = evaluate_compliance(doc_data)
    return jsonify(
        {
            "success": True,
            "data": doc_data.model_dump(),
            "compliance": compliance,
        }
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
