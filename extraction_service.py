"""
Extraction service: uses the OpenAI structured-output API to convert raw
document text into a validated DocumentData Pydantic model.
"""

import logging
import os
from typing import Optional

from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from models import DocumentData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default model – gpt-4o-mini gives a good balance of cost and accuracy.
# Override by setting the OPENAI_MODEL environment variable.
_DEFAULT_MODEL = "gpt-4o-mini"

# System prompt – instructs the model to behave as a strict Procurement and
# Finance reviewer and to return None for any field it cannot confidently
# extract rather than guessing or hallucinating values.
_SYSTEM_PROMPT = (
    "You are a strict Procurement and Finance document reviewer. "
    "Your task is to extract structured data from raw invoice or quotation text "
    "with maximum precision and accuracy.\n\n"
    "Rules:\n"
    "- Be precise and accurate. Do NOT guess or infer values that are not "
    "explicitly present in the text.\n"
    "- If a field cannot be found or extracted with full confidence, return null "
    "for that field. Accuracy is more important than completeness.\n"
    "- Format all dates as YYYY-MM-DD (e.g. 2024-04-15).\n"
    "- Express currency as a 3-letter ISO 4217 code (e.g. USD, EUR, GBP). "
    "Derive the code from symbols ($→USD, €→EUR, £→GBP) only when the context "
    "makes the currency unambiguous.\n"
    "- Return numeric amounts as plain numbers without currency symbols or "
    "thousand-separator commas (e.g. 1000.00, not $1,000.00).\n"
    "- Return tax / discount rates as percentages (e.g. 20.0 for 20 %).\n"
    "- For document_type use lowercase strings such as 'invoice', 'quotation', "
    "'purchase_order', 'credit_note', etc.\n"
    "- Extract all line items when present.\n"
)


def extract_data_from_text(
    raw_text: str,
    temperature: float = 0.0,
    model: Optional[str] = None,
) -> DocumentData:
    """
    Extract structured invoice / quotation data from raw text using the
    OpenAI structured-output (parse) API.

    Parameters
    ----------
    raw_text : str
        The raw text extracted from a document (PDF, DOCX, XLSX, …).
    temperature : float, optional
        Sampling temperature for the model (default 0.0 for deterministic
        output).
    model : str, optional
        OpenAI model to use.  Defaults to the ``OPENAI_MODEL`` environment
        variable, or ``gpt-4o-mini`` if the variable is not set.

    Returns
    -------
    DocumentData
        Pydantic model populated with the fields that could be confidently
        extracted.  Any field that could not be found in the text will be
        ``None``.

    Raises
    ------
    ValueError
        If ``raw_text`` is empty or if the ``OPENAI_API_KEY`` environment
        variable is not set.
    AuthenticationError
        If the API key is invalid or revoked.
    APIConnectionError
        If the network connection to the OpenAI API fails.
    RateLimitError
        If the request is rejected due to rate limiting.
    APIError
        For any other OpenAI API-level error.

    Examples
    --------
    >>> raw_text = \"""
    ... INVOICE
    ... Vendor: Acme Corp
    ... Tax ID: VAT123456789
    ... Invoice #: INV-2024-001
    ... Date: 2024-04-15
    ... Total: $1,000.00 USD
    ... \"""
    >>> result = extract_data_from_text(raw_text)
    >>> result.vendor_name
    'Acme Corp'
    >>> result.total_amount
    1000.0
    """
    # ------------------------------------------------------------------ #
    # Input validation
    # ------------------------------------------------------------------ #
    if not raw_text or not raw_text.strip():
        raise ValueError("raw_text must not be empty.")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. "
            "Set the OPENAI_API_KEY environment variable."
        )

    selected_model = model or os.environ.get("OPENAI_MODEL", _DEFAULT_MODEL)

    # ------------------------------------------------------------------ #
    # Call the OpenAI API
    # ------------------------------------------------------------------ #
    client = OpenAI(api_key=api_key)

    user_message = (
        "Please extract and structure all available information from the "
        "following document text:\n\n"
        f"{raw_text}"
    )

    try:
        logger.debug(
            "Calling OpenAI model '%s' to extract document data "
            "(%d characters of input text).",
            selected_model,
            len(raw_text),
        )

        response = client.beta.chat.completions.parse(
            model=selected_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format=DocumentData,
            temperature=temperature,
        )

        document_data: DocumentData = response.choices[0].message.parsed

        logger.info(
            "Successfully extracted document data using model '%s'.",
            selected_model,
        )
        return document_data

    except AuthenticationError as exc:
        logger.error("OpenAI authentication failed: %s", exc)
        raise AuthenticationError(
            message=(
                "OpenAI authentication failed. "
                "Please check that your OPENAI_API_KEY is valid."
            ),
            response=exc.response,
            body=exc.body,
        ) from exc

    except APIConnectionError as exc:
        logger.error("Could not connect to the OpenAI API: %s", exc)
        raise APIConnectionError(
            message=(
                "Could not connect to the OpenAI API. "
                "Please check your network connection and try again."
            ),
            request=exc.request,
        ) from exc

    except RateLimitError as exc:
        logger.error("OpenAI rate limit exceeded: %s", exc)
        raise RateLimitError(
            message=(
                "OpenAI rate limit exceeded. "
                "Please wait a moment before retrying."
            ),
            response=exc.response,
            body=exc.body,
        ) from exc

    except APIError as exc:
        logger.error("OpenAI API error: %s", exc)
        raise APIError(
            message=f"OpenAI API error: {exc.message}",
            request=exc.request,
            body=exc.body,
        ) from exc
