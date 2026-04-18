def extract_text_from_pdf(pdf_file):
    import pytesseract
    from pdf2image import convert_from_path

    # First, try extracting text normally
    text = extract_text(pdf_file)  # Assumes an existing function to extract text

    # Fallback for OCR if text is less than 50 characters
    if len(text) < 50:
        # Convert PDF to images
        images = convert_from_path(pdf_file)
        # Use pytesseract to do OCR on each image
        ocr_text = ''
        for image in images:
            ocr_text += pytesseract.image_to_string(image)
        return ocr_text
    return text