from pathlib import Path
import pandas as pd


def load_zuco2_data():
    """
    Load ZuCo 2.0 sentences data as df.
    """
    base_root = Path(__file__).parent.parent.parent / "zuco"
    zuco_sentences_file = base_root / "zuco2_sentences.csv"
    df_zuco = pd.read_csv(zuco_sentences_file, sep='\t')
    print(f"ZuCo 2.0 Sentences: {len(df_zuco)}")
    return df_zuco
