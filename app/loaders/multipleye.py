import pandas as pd
from pathlib import Path


def get_txt_files_as_df(folder_paths):
    data_list = []
    for folder in folder_paths:
        path_obj = Path(folder)
        if path_obj.exists():
            for file_path in path_obj.rglob('*.txt'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    data_list.append({
                        'filename': file_path.name,
                        'subcategory': file_path.parent.name,
                        'text': content
                    })
                except Exception as e:
                    print(f"Skipping {file_path.name}: {e}")
        else:
            print(f"Warning: Path not found - {folder}")

    df = pd.DataFrame(data_list)
    if df.empty: return df
    return df[df['text'].str.strip() != '']


def load_multipleye_data():
    base_root = Path(__file__).parent.parent.parent / "multipleye"

    ro_path = [f"{base_root}/ro/texts_updated"]
    en_path = [f"{base_root}/en"]

    df_ro = get_txt_files_as_df(ro_path)
    df_en_raw = get_txt_files_as_df(en_path)

    df_en = (df_en_raw
             .sort_values('filename')
             .groupby('subcategory', as_index=False)['text']
             .apply(lambda x: ' '.join(x))
    )

    print(f"Romanian Texts: {len(df_ro)}")
    print(f"English Texts:  {len(df_en)}")

    return df_ro, df_en
