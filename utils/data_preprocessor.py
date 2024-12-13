def clean_text(text):
    """Remove unwanted characters from the text."""
    text = text.replace("\n", " ")
    return text.strip()
