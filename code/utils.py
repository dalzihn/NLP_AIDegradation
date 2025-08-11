import re

import pandas as pd

from helper import *


def concat_data(data_paths: list[str]) -> pd.DataFrame:
    """Concatenate multiple csv files into a single DataFrame

    Args:
        data_paths: list of csv file paths
    Returns:
        pd.DataFrame: concatenated DataFrame"""
    new_df = pd.concat([pd.read_csv(path) for path in data_paths], ignore_index=True)
    return new_df


def label_mapping(df: pd.DataFrame, stars_column="stars") -> pd.Series:
    """
    Map star ratings in text format to Vietnamese sentiment labels in a pandas DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing star ratings
        stars_column (str): Name of the column containing star ratings in text format (default: 'stars')

    Returns:
        pd.Series with a new mapping
    """

    def map_stars(stars_text):
        if pd.isna(stars_text) or not isinstance(stars_text, str):
            return None

        # Clean and extract the number from the text
        try:
            # Remove any non-digit characters and convert to integer
            star_count = int("".join(filter(str.isdigit, stars_text)))
        except (ValueError, IndexError):
            return None

        # Map to sentiment labels
        if star_count >= 4:
            return "Tích cực"
        elif star_count == 3:
            return "Trung tính"
        elif star_count <= 2:
            return "Tiêu cực"
        return None

    return df[stars_column].apply(map_stars)


def expand_vietnamese_acronyms(text: str) -> str:
    """
    Expand Vietnamese internet acronyms in the given text.

    Args:
        text (str): Input text containing Vietnamese acronyms

    Returns:
        str: Text with acronyms expanded to full form
    """
    if not text:
        return text

    words = text.split()
    expanded_words = []

    for word in words:
        # Convert to lowercase for case-insensitive matching
        lower_word = word.lower()

        # Check if word is in our dictionary
        if lower_word in VIETNAMESE_ACRONYMS:
            # Preserve original capitalization if the word was capitalized
            if word[0].isupper():
                expanded = VIETNAMESE_ACRONYMS[lower_word].capitalize()
            else:
                expanded = VIETNAMESE_ACRONYMS[lower_word]
            expanded_words.append(expanded)
        else:
            expanded_words.append(word)

    return " ".join(expanded_words)


def normalise_elongated_text(text: str) -> str:
    """Normalise elongated text

    Args:
        text (str): Input of elongated text

    Returns:
        str: Normalised text
    """
    normalised_text = re.sub(r"(.)\1{2,}", r"\1", text)
    return normalised_text
