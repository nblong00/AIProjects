
def clean_api_text(text: str) -> str:
    text = text.replace("/n", "\n")
    text = text.replace("\\n", "\n")
    try:
        text = text.encode("utf-8").decode("unicode_escape")
    except Exception:
        pass
    return text

