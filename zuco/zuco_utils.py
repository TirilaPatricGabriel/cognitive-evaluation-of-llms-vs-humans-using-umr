import pandas as pd


def create_zuco_sentences_file(zuco_eye_file, zuco_sentences_file):
    """
    Create a sentences file from the ZuCo eye-tracking data file.
    """

    df = pd.read_csv(zuco_eye_file, sep=',')
    sentences = df[['sentence_id', 'sentence']].drop_duplicates().sort_values('sentence_id')
    sentences.to_csv(zuco_sentences_file, sep='\t', index=False)
        

def create_average_subject_file(zuco_eye_file, output_file):
    """
    Create an averaged-subject eye-tracking file from ZuCo data.
    Averages eye-tracking metrics across subjects per word.
    """

    df = pd.read_csv(zuco_eye_file, sep=",")

    eye_metrics = ["FFD", "GD", "GPT", "TRT", "nFix", "reading_order"]

    grouped = df.groupby(
        ["sentence", "sentence_id", "word_id", "content", "task", "freq"],
        as_index=False
    )[eye_metrics].mean()

    grouped = grouped.rename(columns={m: f"{m}_avg" for m in eye_metrics})
    grouped = grouped.round(2)
    grouped = grouped.sort_values(["sentence_id", "word_id"])

    grouped.to_csv(output_file, index=False)
    


def create_average_sentence_level_file(word_level_avg_file: str, output_file: str):
    """
    Aggregate word-level averaged eye-tracking metrics into sentence-level measures.
    """

    df = pd.read_csv(word_level_avg_file)

    agg = df.groupby("sentence_id").agg(
        FFD_avg=("FFD_avg", "mean"),
        GD_avg=("GD_avg", "mean"),
        GPT_avg=("GPT_avg", "mean"),

        TRT_avg=("TRT_avg", "mean"),
        TRT_sum=("TRT_avg", "sum"),

        nFix_avg=("nFix_avg", "mean"),
        nFix_sum=("nFix_avg", "sum"),

        reading_order_avg=("reading_order_avg", "mean")
    ).reset_index()


    agg = agg.round(2)

    agg.to_csv(output_file, index=False)
    print(f"Saved sentence-level eye-tracking file to {output_file}")
    
    
def create_participants_sentence_level_file(zuco_eye_file: str, output_file: str):
    """
    Create sentence-level eye-tracking file for each participant from ZuCo data.
    Compute for each participant the sentence-level averages and sums and save to the same output file.
    """
    
    df = pd.read_csv(zuco_eye_file, sep=",")

    participant_dfs = []
    for subject_id, group in df.groupby("subject"):
        agg = group.groupby("sentence_id").agg(
            FFD_avg=("FFD", "mean"),
            GD_avg=("GD", "mean"),
            GPT_avg=("GPT", "mean"),

            TRT_avg=("TRT", "mean"),
            TRT_sum=("TRT", "sum"),

            nFix_avg=("nFix", "mean"),
            nFix_sum=("nFix", "sum"),

            reading_order_avg=("reading_order", "mean")
        ).reset_index()
        agg['subject'] = subject_id
        participant_dfs.append(agg)

    result_df = pd.concat(participant_dfs, ignore_index=True)
    result_df = result_df.round(2)

    result_df.to_csv(output_file, index=False)
    print(f"Saved participant-level sentence eye-tracking file to {output_file}")
    
    
def create_participants_word_level_file(zuco_eye_file: str, output_file: str):
    """
    Create word-level eye-tracking file for each participant from ZuCo data.
    Save to the same output file.
    """
    
    df = pd.read_csv(zuco_eye_file, sep=",")

    # Select relevant columns
    eye_metrics = ["FFD", "GD", "GPT", "TRT", "nFix", "reading_order"]
    selected_columns = ["subject", "sentence_id", "word_id", "content"] + eye_metrics
    participant_word_level_df = df[selected_columns].copy()
    participant_word_level_df.to_csv(output_file, index=False)
    print(f"Saved participant-level word eye-tracking file to {output_file}")
    
    
def create_average_word_level_file(zuco_eye_file: str, output_file: str):
    """
    Create word-level averaged eye-tracking file across participants from ZuCo data.
    Save to the same output file.
    """
    
    df = pd.read_csv(zuco_eye_file, sep=",")

    eye_metrics = ["FFD", "GD", "GPT", "TRT", "nFix", "reading_order"]

    grouped = df.groupby(
        ["sentence_id", "word_id", "content"],
        as_index=False
    )[eye_metrics].mean()

    grouped = grouped.round(2)
    grouped = grouped.sort_values(["sentence_id", "word_id"])

    grouped.to_csv(output_file, index=False)
    print(f"Saved averaged word-level eye-tracking file to {output_file}")
