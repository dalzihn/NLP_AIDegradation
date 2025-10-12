from pathlib import Path

import emoji
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split

# Dictionary of common Vietnamese internet slang and acronyms
VIETNAMESE_ACRONYMS = {
    "ko": "không",
    "k": "không",
    "kh": "không",
    "kg": "không",
    "dc": "được",
    "đc": "được",
    "bt": "bình thường",
    "thjk": "thích",
    "thik": "thích",
    "thjk": "thích",
    "ng": "người",
    "ngta": "người ta",
    "t": "tôi",
    "tui": "tôi",
    "dth": "dễ thương",
    "đx": "được",
    "trl": "trả lời",
    "nv": "nhân viên",
    "nvien": "nhân viên",
    "sp": "sản phẩm",
    "pvu": "phục vụ",
    "r": "rồi",
    "ròi": "rồi",
    "z": "vậy",
    "v": "vậy",
    "hok": "không",
    "ik": "đi",
    "e": "em",
    "đt": "điện thoại",
    "dt": "điện thoại",
    "dthoai": "điện thoại",
    "mk": "mình",
    "sd": "sử dụng",
    "r": "rồi",
    "luông": "luôn",
    "sp": "sản phẩm",
    "zậy": "vậy",
    "vd": "ví dụ",
    "b": "bạn",
    "m": "mình",
    "dk": "được",
    "cx": "cũng",
    "ak": "à",
    "jo": "giờ",
    "ad": "à",
    "t": "tôi",
    "dị": "vậy",
    "h": "giờ",
    "kb": "không biết",
    "kbt": "không biết",
    "khongbiet": "không biết",
    "khoinghira": "không nghĩ ra",
    "khonghieu": "không hiểu",
    "hokbit": "không biết",
    "hokpit": "không biết",
    "bitroi": "biết rồi",
    "bikroi": "biết rồi",
    "ck": "chồng",
    "đỉm": "điểm",
    "vk": "vợ",
    "ny": "người yêu",
    "lm": "làm",
    "lm j": "làm gì",
    "lm sao": "làm sao",
    "j": "gì",
    "kcj": "không có gì",
    "gòi": "rồi",
    "ròi": "rồi",
    "rùi": "rồi",
    "iu": "yêu",
    "iuu": "yêu",
    "iu nhìu": "yêu nhiều",
    "nhìu": "nhiều",
    "nhju": "nhiều",
    "nhìu wa": "nhiều quá",
    "wa": "quá",
    "wá": "quá",
    "wua": "qua",
    "wê": "quê",
    "wên": "quên",
}


def concat_results(
    num_files: int,
    dir: Path,
    prefix: str,
    file_pattern: list[str],
    extra_col_value: tuple[str, str],
) -> pd.DataFrame:
    """
    Concatenate results from multiple CSV files into a single DataFrame.

    Args:
     num_files: The number of files to be concatenated.
     dir: Path to the directory storing results.
     prefix: The prefix of the names of the files to concatenate.
     pattern: A list of strings representing patterns to look for within the filenames.
     extra_col_value: An optional value to add as a new column will be used as hue for plotting.

    Returns:
     A pandas DataFrame containing the concatenated data.
    """
    list_df = [
        pd.read_csv(dir / f"{prefix}{pattern}.csv").tail(
            1
        )  # Get the final training results
        for pattern in file_pattern
    ]

    df_concat = pd.concat(list_df)
    df_concat[extra_col_value[0]] = extra_col_value[1]

    df_concat["generation"] = ["Real data"] + [str(i) for i in range(1, num_files + 1)]

    df_concat.reset_index(drop=True, inplace=True)

    df_concat.drop(["Unnamed: 0"], axis=1, inplace=True)

    df_concat = round(df_concat, 4)

    return df_concat


def draw_lines(
    data: pd.DataFrame, hue: str, cols_to_plot: list[str], n_cols: int, n_rows: int
):
    """
    Draws multiple line plots from a list of pandas DataFrames.

    Args:
      data: A pandas DataFrame containing the data to plot.
      hue: The column name to use for the hue (e.g., 'model').
      cols_to_plot: A list of column names to plot on separate subplots.
      n_cols: The number of columns for the subplot grid.
      n_rows: The number of rows for the subplot grid.
    """
    # -  plt.style.use('seaborn-v0_8-whitegrid') # Using a different style for better aesthetics
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(25, 10)
    )  # Increased figure width for better x-axis spacing
    sns.set_style("ticks")

    # Flatten the axes array for easy iteration
    axs = axs.flatten()

    # Get unique generation values for x-axis ticks
    generation_ticks = data["generation"].unique()

    # Draw line plot for each column to plot
    for i, col in enumerate(cols_to_plot):
        ax = axs[i]
        sns.lineplot(
            data=data,
            x="generation",
            y=col,
            ax=ax,
            palette="husl",
            hue=hue,
            linewidth=2,  # Increase line thickness
        )

        ax.set_xlabel("")
        ax.set_ylabel("")
        # Remove "Test" from the title and capitalize
        title = col.replace("_", " ").title().replace("Test ", "")
        ax.set_title(title, fontsize=14)  # Capitalize and space out title
        # ax.legend(legends)

        # Set x-axis ticks and rotate labels
        ax.set_xticks(generation_ticks)
        # ax.set_yticks(np.arange(data[col].min()-0.05, data[col].max()-0.05, 0.05))
        ax.tick_params(axis="x", rotation=45)

        # Add grid lines for better readability
        # ax.grid(True, linestyle='--', alpha=0.6)

    # Remove any unused subplots
    for j in range(len(cols_to_plot), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()


def label_mapping(df: pd.DataFrame, stars_column="stars") -> pd.Series:
    """
    Map star ratings in text format to Vietnamese sentiment labels in a pandas DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing star ratings
        stars_column (str): Name of the column containing star ratings in text format (default: 'stars')

    Returns:
        pd.Series with a new mapping
    """

    def map_stars(stars_count):
        if pd.isna(stars_count) or not isinstance(stars_count, int):
            return None

        # Map to sentiment labels
        if stars_count >= 4:
            return 2
        elif stars_count == 3:
            return 1
        elif stars_count <= 2:
            return 0
        return None

    return df[stars_column].apply(map_stars)


def split_data(
    df: pd.DataFrame,
    firsttrain_frac: float = 0.5,
    test_frac: float = 0.1,
    random_state: int = 42,
    n_splits: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    """Splits a DataFrame into training, testing, and k-fold cross-validation sets.

    The data is first split into a temporary training set and a test set.
    The temporary training set is then split into the final training set
    and a set for generating folds. StratifiedKFold is used
    to create the specified number of folds from this set, ensuring
    stratification based on the 'sentiment' column.

    Args:
        df: The input DataFrame to split.
        firsttrain_frac: The fraction of the original DataFrame to use as the training set.
        test_frac: The fraction of the original DataFrame to use as the test set.
        random_state: The seed for random number generation for reproducibility.
        n_splits: The number of folds to get.

    Returns:
        A tuple containing three elements:
        - train_df (pd.DataFrame): The main training set.
        - test_df (pd.DataFrame): The test set.
        - folds (list[pd.DataFrame]): A list of DataFrames, where each DataFrame
                                     represents a unique fold (test set) for
                                     cross-validation.
    """
    df = df.sample(frac=1, random_state=random_state)  # Shuffle the DataFrame
    predict_label_size = 1 - (firsttrain_frac / (1 - test_frac))

    train_tmp_df, test_df = train_test_split(
        df, test_size=test_frac, random_state=random_state
    )

    train_df, model_labels_df = train_test_split(
        train_tmp_df, test_size=predict_label_size, random_state=random_state
    )

    skf = StratifiedKFold(n_splits=n_splits)
    folds = []
    for train_index, test_index in skf.split(
        model_labels_df, model_labels_df["sentiment"]
    ):
        folds.append(
            model_labels_df.iloc[test_index]
        )  # Return the test set for each fold

    return train_df, test_df, folds


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


def clean_emojis(text: str) -> str:
    """Remove emojis from text

    Args:
        text (str): Input text containing emojis
    Returns:
        str: Text with emojis removed
    """
    # Remove emojis
    text_no_emoji = emoji.replace_emoji(text, replace="")

    # Remove extra whitespaces
    text_no_emoji = re.sub(r"\s+", " ", text_no_emoji).strip()

    return text_no_emoji


def clean_text(text: str) -> str:
    """Clean text by
    1. Remove extra whitespaces
    2. Expand acronyms
    3. Replace elongated words

    Args:
        text (str): Text to clean

    Returns:
        str: Cleaned text
    """
    # Remove extra whitespaces
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    # Expand acronyms
    text = expand_vietnamese_acronyms(text)

    # Replace elongated words
    text = normalise_elongated_text(text)

    # Translate emojis and emoticons
    text = clean_emojis(text)

    return text


def preprocess(
    df: pd.DataFrame, model_name: str, dataset_name: str, col: str = "text"
) -> pd.DataFrame:
    if (
        dataset_name == "customer_feedback"
        or dataset_name == "vietnamese_sentiment"
        or dataset_name == "concat"
    ):
        df[col] = df[col].map(lambda x: x.lower())
        df[col] = df[col].map(lambda x: clean_text(x))

    if model_name == "phobert":
        df[col] = df[col].map(lambda x: " ".join(rdrsegmenter.word_segment(x)))

    return df
