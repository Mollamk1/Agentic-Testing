import pdfplumber
import pandas as pd
from docx import Document


def extract_text_from_file(file_path: str) -> str:
    """
    Extracts text from a file based on its extension.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The extracted text from the file.

    Raises:
        ValueError: If the file extension is not supported.
    """
    extension = file_path.split('.')[-1].lower()

    if extension == 'pdf':
        return extract_text_from_pdf(file_path)
    elif extension == 'docx':
        return extract_text_from_docx(file_path)
    elif extension == 'xlsx':
        return extract_text_from_xlsx(file_path)
    else:
        raise ValueError("Unsupported file type. Please use PDF, DOCX, or XLSX.")


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text.
    """
    with pdfplumber.open(file_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    return text


def extract_text_from_docx(file_path: str) -> str:
    """
    Extracts text from a DOCX file.

    Args:
        file_path (str): The path to the DOCX file.

    Returns:
        str: The extracted text.
    """
    doc = Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])


def extract_text_from_xlsx(file_path: str) -> str:
    """
    Extracts text from an XLSX file.

    Args:
        file_path (str): The path to the XLSX file.

    Returns:
        str: The extracted text.
    """
    df = pd.read_excel(file_path)
    return df.to_string(index=False)


if __name__ == '__main__':
    file_path = 'path/to/your/file'  # example usage
    try:
        text = extract_text_from_file(file_path)
        print(text)
    except ValueError as e:
        print(e)
