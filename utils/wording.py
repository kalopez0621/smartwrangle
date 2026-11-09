"""
utils/wording.py
================
Small helpers that keep on-screen text natural.

plural()   "1 value" / "3 values" instead of "3 value(s)"
"""


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    """
    Write a count with the right form of the word after it.

    Examples
    --------
    plural(1, "value")            -> "1 value"
    plural(1240, "value")         -> "1,240 values"
    plural(2, "category", "categories") -> "2 categories"

    Parameters
    ----------
    count       : int   How many there are.
    singular    : str   The word for one ("row", "value").
    plural_form : str   The word for more than one. Leave it out to add "s".
    """
    # Default plural: add "s" (row -> rows). Pass plural_form for words like
    # "category" -> "categories".
    word = singular if count == 1 else (plural_form or singular + "s")
    # :, puts commas in big numbers (1,240)
    return f"{count:,} {word}"
