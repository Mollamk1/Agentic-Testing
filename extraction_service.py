"""
Extraction service for invoice and quotation data.

Uses OpenAI's structured output API (client.beta.chat.completions.parse) to
extract structured invoice/quotation data from raw text with strict validation.

Example usage::

    text = "Invoice from Acme Corp, Invoice #INV-001, Total $500 USD"
    data = extract_data_from_text(text)
    print(data.vendor_name)          # "Acme Corp"
    print(data.total_gross_amount)   # 500.0
"""

import logging
import os
from typing import List, Optional

from openai import (
    AuthenticationError,
    APIConnectionError,
    RateLimitError,
    APIError,
    OpenAI,
)
from pydantic import BaseModel, ValidationError

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Data Models
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are a Strict Procurement and Finance Reviewer.

Your task is to extract structured invoice and quotation data from the provided document text.

Rules:
- Only extract data that is EXPLICITLY present in the document.
- Do NOT guess, infer, or fabricate any values.
- Return null/None for any field that is not clearly present in the text.
- For dates, use YYYY-MM-DD format (e.g., 2024-01-15). Return null if not found.
- For currency, use 3-letter ISO 4217 codes (USD, EUR, GBP, etc.). Return null if not found.
- For numeric amounts, use decimal format without currency symbols (e.g., 1500.00). Return null if not found.
- All extracted values must be accurate and directly verifiable from the source text.
"""


class LineItem(BaseModel):
    """Represents a single line item in an invoice or quotation."""

    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total_price: Optional[float] = None


class DocumentData(BaseModel):
    """Structured data extracted from an invoice or quotation document.

    All fields are optional and will be None if not explicitly present in the
    source text.

    Example::

        data = DocumentData(
            vendor_name="Acme Corp",
            document_number="INV-001",
            total_gross_amount=500.0,
            currency="USD",
        )
    """

    document_type: Optional[str] = None
    """Type of document, e.g. 'invoice' or 'quotation'."""

    document_number: Optional[str] = None
    """Invoice or quotation reference number, e.g. 'INV-001'."""

    document_date: Optional[str] = None
    """Date the document was issued, in YYYY-MM-DD format."""

    due_date: Optional[str] = None
    """Payment due date, in YYYY-MM-DD format."""

    vendor_name: Optional[str] = None
    """Name of the vendor or supplier issuing the document."""

    vendor_address: Optional[str] = None
    """Full address of the vendor or supplier."""

    customer_name: Optional[str] = None
    """Name of the customer or buyer."""

    customer_address: Optional[str] = None
    """Full address of the customer or buyer."""

    line_items: Optional[List[LineItem]] = None
    """List of individual line items (products or services)."""

    subtotal: Optional[float] = None
    """Pre-tax subtotal amount as a decimal number."""

    tax_amount: Optional[float] = None
    """Total tax amount as a decimal number."""

    total_gross_amount: Optional[float] = None
    """Final total amount (including tax) as a decimal number."""

    currency: Optional[str] = None
    """ISO 4217 currency code, e.g. 'USD', 'EUR', 'GBP'."""

    payment_terms: Optional[str] = None
    """Payment terms, e.g. 'Net 30', 'Due on receipt'."""

    notes: Optional[str] = None
    """Any additional notes or comments from the document."""


# --------------------------------------------------------------------------- #
# Extraction Function
# --------------------------------------------------------------------------- #


def extract_data_from_text(raw_text: str) -> DocumentData:
    """Extract structured invoice/quotation data from raw text using OpenAI structured output.

    Sends the provided text to the OpenAI API using the structured output
    feature (``client.beta.chat.completions.parse``) and returns a validated
    :class:`DocumentData` instance.

    Args:
        raw_text: The raw text content of an invoice or quotation document.

    Returns:
        A :class:`DocumentData` instance with all fields populated from the
        document.  Any field not explicitly present in the source text will be
        ``None``.

    Raises:
        ValueError: If ``raw_text`` is empty or ``None``.
        AuthenticationError: If the OpenAI API key is invalid or missing.
        APIConnectionError: If there is a network connectivity issue.
        RateLimitError: If the OpenAI API rate limit has been exceeded.
        APIError: For any other OpenAI API error.
        ValidationError: If the API response cannot be validated against the
            :class:`DocumentData` schema.

    Example::

        text = "Invoice from Acme Corp, Invoice #INV-001, Total $500 USD"
        data = extract_data_from_text(text)
        print(data.vendor_name)         # "Acme Corp"
        print(data.total_gross_amount)  # 500.0
    """
    # ---- 1. Validate input ------------------------------------------------ #
    if not raw_text or not raw_text.strip():
        raise ValueError("raw_text must be a non-empty string.")

    # ---- 2. Initialise OpenAI client -------------------------------------- #
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    logger.info("Sending %d characters to OpenAI for extraction.", len(raw_text))

    # ---- 3. Call OpenAI structured output API ----------------------------- #
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            response_format=DocumentData,
        )

        document_data: DocumentData = completion.choices[0].message.parsed

        if document_data is None:
            raise ValueError("OpenAI returned a null response instead of DocumentData.")

        logger.info(
            "Extraction successful. vendor_name=%r, total_gross_amount=%r",
            document_data.vendor_name,
            document_data.total_gross_amount,
        )
        return document_data

    except AuthenticationError as exc:
        logger.error("Authentication failed: invalid or missing OPENAI_API_KEY. %s", exc)
        raise

    except APIConnectionError as exc:
        logger.error(
            "Network error: could not connect to OpenAI API. "
            "Check your internet connection. %s",
            exc,
        )
        raise

    except RateLimitError as exc:
        logger.error(
            "Rate limit exceeded: OpenAI API quota has been reached. "
            "Please wait before retrying. %s",
            exc,
        )
        raise

    except APIError as exc:
        logger.error("OpenAI API error: %s", exc)
        raise

    except ValidationError as exc:
        logger.error("Schema validation error: the API response did not match DocumentData. %s", exc)
        raise
