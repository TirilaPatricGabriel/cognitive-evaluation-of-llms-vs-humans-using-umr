import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import statsmodels.formula.api as smf
import itertools
import re
import string
from nltk.stem import WordNetLemmatizer


class UMREyeTrackingComparator:
    """
    Compare eye-tracking data with UMR data at sentence-level.
    """

    def __init__(self, df_umr_sent: pd.DataFrame, df_eye_sent_avg: pd.DataFrame, df_eye_sent_participants: pd.DataFrame = None, df_umr_node: pd.DataFrame = None, df_eye_word_avg = None, df_eye_word_participants: pd.DataFrame = None):
        self.df_umr_sent = df_umr_sent
        self.df_eye_sent_avg = df_eye_sent_avg
        self.df_eye_sent_participants = df_eye_sent_participants
        self.df_umr_node = df_umr_node
        self.df_eye_word_avg = df_eye_word_avg
        self.df_eye_word_participants = df_eye_word_participants
        self.lemmatizer = WordNetLemmatizer()

        # Merge average data
        self.df_sent_avg_merged = pd.merge( df_eye_sent_avg, df_umr_sent, on="sentence_id", how="inner")

        # Merge participant data
        if df_eye_sent_participants is not None:
            self.df_sent_participant_merged = pd.merge(df_eye_sent_participants, df_umr_sent, on="sentence_id", how="inner")
        else:
            self.df_sent_participant_merged = None


    def compute_correlations(
        self,
        umr_features=None,
        eye_metrics=None,
        return_heatmap=False,
        participant_level=False,
        level="sentence"  # "sentence" or "word"
    ):
        """
        Compute correlations between UMR features and eye-tracking metrics.
        
        Args:
            umr_features: list of UMR features to include
            eye_metrics: list of eye-tracking metrics
            return_heatmap: if True, return base64-encoded heatmap
            participant_level: if True, compute correlations per participant and summarize
            level: "sentence" or "word"
        
        Returns:
            dict with sorted correlations, correlation matrix, heatmap (base64), participant-level summary
        """

        # Select appropriate dataframe
        if level == "sentence":
            df_to_use = self.df_sent_participant_merged if participant_level else self.df_sent_avg_merged
            if umr_features is None:
                umr_features = [
                    "num_nodes", "num_edges", "max_depth", "avg_depth",
                    "num_predicates", "num_entities", "predicate_entity_ratio",
                    "num_reentrancies", "avg_degree", "max_degree",
                    "num_coordination", "num_temporal_quantities"
                ]
            if eye_metrics is None:
                eye_metrics = ["FFD_avg", "GD_avg", "GPT_avg", "TRT_avg", "TRT_sum", "nFix_avg", "nFix_sum", "reading_order_avg"]
        elif level == "word":
            if umr_features is None:
                umr_features = ["depth", "degree", "in_degree", "out_degree"]
            if eye_metrics is None:
                eye_metrics = ["FFD", "GD", "GPT", "TRT", "nFix", "reading_order"]
            if participant_level:
                if not hasattr(self, 'df_merged_word_participants'):
                    raise ValueError("Word-level participant-level merged dataframe not available.")
                df_to_use = self.df_merged_word_participants
            else:
                if not hasattr(self, 'df_merged_word_avg'):
                    raise ValueError("Word-level average merged dataframe not available.")
                df_to_use = self.df_merged_word_avg
        else:
            raise ValueError("Invalid level. Choose 'sentence' or 'word'.")

        if df_to_use is None or df_to_use.empty:
            raise ValueError(f"No data available for level '{level}'.")

        # Overall correlations
        correlations = []
        for umr_feat in umr_features:
            for eye_metric in eye_metrics:
                if umr_feat in df_to_use.columns and eye_metric in df_to_use.columns:
                    corr_val = df_to_use[umr_feat].corr(df_to_use[eye_metric])
                    if pd.isna(corr_val) or np.isinf(corr_val):
                        corr_val = 0
                    correlations.append({
                        "UMR_feature": umr_feat,
                        "Eye_metric": eye_metric,
                        "Correlation": round(corr_val, 2)
                    })

        correlations_df = pd.DataFrame(correlations)
        correlations_sorted = correlations_df.reindex(correlations_df["Correlation"].abs().sort_values(ascending=False).index).reset_index(drop=True)

        # Correlation matrix
        corr_matrix = df_to_use[[f for f in umr_features if f in df_to_use.columns] + [e for e in eye_metrics if e in df_to_use.columns]].corr().loc[umr_features, eye_metrics]
        corr_matrix = corr_matrix.fillna(0).replace([np.inf, -np.inf], 0)
        corr_matrix_json = corr_matrix.round(2).to_dict()

        # Heatmap
        heatmap_base64 = None
        if return_heatmap:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
            plt.title(f"Correlation: UMR features vs Eye-tracking metrics ({level}-level)")
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            heatmap_base64 = base64.b64encode(buf.read()).decode("utf-8")
            with open(f"plots/umr_eye_corr_heatmap_{level}.png", "wb") as f:
                f.write(base64.b64decode(heatmap_base64))

        # Participant-level correlations
        participant_corrs_summary = None
        if participant_level:
            all_corrs = []
            for subj in df_to_use['subject'].unique():
                df_subj = df_to_use[df_to_use['subject'] == subj]
                for umr_feat in umr_features:
                    for eye_metric in eye_metrics:
                        if umr_feat in df_subj.columns and eye_metric in df_subj.columns:
                            corr_val = df_subj[umr_feat].corr(df_subj[eye_metric])
                            if pd.isna(corr_val) or np.isinf(corr_val):
                                corr_val = 0
                            all_corrs.append({
                                "subject": subj,
                                "UMR_feature": umr_feat,
                                "Eye_metric": eye_metric,
                                "Correlation": corr_val
                            })

            df_all = pd.DataFrame(all_corrs)
            participant_corrs_summary = df_all.groupby(['UMR_feature', 'Eye_metric'])['Correlation'].agg(['mean', 'std']).reset_index()
            participant_corrs_summary['mean'] = participant_corrs_summary['mean'].round(2)
            participant_corrs_summary['std'] = participant_corrs_summary['std'].round(2)
            participant_corrs_summary['mean_std'] = participant_corrs_summary['mean'].astype(str) + " ± " + participant_corrs_summary['std'].astype(str)

        return {
            "correlations_sorted": correlations_sorted.to_dict(orient="records"),
            "corr_matrix": corr_matrix_json,
            "heatmap_base64": heatmap_base64,
            "participant_level_summary": participant_corrs_summary.to_dict(orient="records") if participant_corrs_summary is not None else None
        }

        
    def plot_participant_correlations(self, participant_corrs_summary, save_path="plots/participant_corrs.png"):
        """
        Plot participant-level correlations as mean ± SD using dots and vertical segments.
        """
        
        # Convert to DataFrame if needed
        if isinstance(participant_corrs_summary, list):
            participant_corrs_summary = pd.DataFrame(participant_corrs_summary)

        if participant_corrs_summary.empty:
            print("No participant-level correlations to plot.")
            return None

        # Create a unique label for each UMR x Eye pair
        participant_corrs_summary['pair_label'] = participant_corrs_summary['UMR_feature'] + " | " + participant_corrs_summary['Eye_metric']

        fig, ax = plt.subplots(figsize=(16, 6))

        # Plot mean and SD using errorbar
        ax.errorbar(
            x=range(len(participant_corrs_summary)),
            y=participant_corrs_summary['mean'],
            yerr=participant_corrs_summary['std'],
            fmt='o', 
            ecolor='gray', 
            elinewidth=2,
            capsize=4,
            markersize=6,
            color='blue'
        )

        ax.set_xticks(range(len(participant_corrs_summary)))
        ax.set_xticklabels(participant_corrs_summary['pair_label'], rotation=90)
        ax.set_ylabel("Correlation")
        ax.set_title("Participant-level correlations: mean ± SD")
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.tight_layout()

        # Save plot to file + base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        with open(save_path, "wb") as f:
            f.write(base64.b64decode(image_base64))
        print(f"Participant-level correlations bar plot saved to {save_path}")

        return image_base64
    
    
    def run_mixed_effects_sent_level_batch(self, umr_features, eye_metrics, max_features_per_model=2, save_path="results/mixed_effects_results.csv"):
        """
        Run mixed-effects models for multiple UMR feature combinations and eye metrics.
        
        Models:
            eye_metric ~ UMR_features + (1 | subject)
        """

        if self.df_sent_participant_merged is None:
            raise ValueError("Participant-level data is required.")

        results = []

        # Generate feature combinations
        feature_combinations = []
        for k in range(1, max_features_per_model + 1):
            feature_combinations.extend(itertools.combinations(umr_features, k))

        for eye_metric in eye_metrics:
            for feat_combo in feature_combinations:
                cols = ["subject", eye_metric] + list(feat_combo)
                df_model = self.df_sent_participant_merged[cols].dropna()

                # Skip too small datasets
                if df_model["subject"].nunique() < 2 or len(df_model) < 50:
                    continue

                fixed_formula = " + ".join(feat_combo)
                formula = f"{eye_metric} ~ {fixed_formula}"

                try:
                    model = smf.mixedlm(formula, df_model, groups=df_model["subject"])
                    result = model.fit(reml=True, disp=False)
                except Exception as e:
                    print(f"Model failed: {formula}")
                    continue

                # Save fixed effects
                for feat in feat_combo:
                    results.append({
                        "eye_metric": eye_metric,
                        "umr_feature": feat,
                        "feature_set": "+".join(feat_combo),
                        "coef": round(result.params.get(feat, 0), 4),
                        "p_value": round(result.pvalues.get(feat, 1), 6),
                        "intercept": round(result.params.get("Intercept", 0), 4),
                        "random_effect_var": round(result.cov_re.iloc[0, 0], 4),
                        "n_obs": len(df_model),
                        "n_subjects": df_model["subject"].nunique()
                    })

        df_results = pd.DataFrame(results)
        df_results.to_csv(save_path, index=False)
        print(f"Mixed-effects results saved to {save_path}")

        return df_results
    
    
    def normalize_word(self, w):
        """
        Normalize eye-tracking tokens safely:
        - handle NaN / float
        - lowercase
        - strip punctuation
        - lemmatize
        """
        if not isinstance(w, str):
            return ""
        w = w.lower().strip()
        w = w.strip(string.punctuation)

        if not w:
            return ""
        return self.lemmatizer.lemmatize(w)

    
    def merge_word_level_data(self):
        """
        Merge UMR node-level data with participant-level word eye-tracking data.
        Match on sentence_id and normalized word.
        """
        if self.df_umr_node is None or self.df_eye_word_participants is None or self.df_eye_word_avg is None:
            raise ValueError("UMR node-level data and eye-tracking word-level participant and eye-tracking word-level average data are required.")

        # Go through eye tracking data
        df_participants_merged = []
        df_avg_merged = []
        for df_eye in [self.df_eye_word_participants, self.df_eye_word_avg]:
            merged_rows = []
            for _, eye_row in df_eye.iterrows():
                sent_id = eye_row["sentence_id"]
                word = self.normalize_word(eye_row["content"])
                if not word:
                    continue
                # Find matching UMR node
                # Get the nodes for this sentence
                umr_nodes_sent = self.df_umr_node[self.df_umr_node["sentence_id"] == sent_id]
                for _, umr_row in umr_nodes_sent.iterrows():
                    umr_word = umr_row["word"]
                    # For predicates, remove suffixes like "-01"
                    umr_word_norm = self.normalize_word(re.sub(r"-\d+$", "", umr_word))
                    if word == umr_word_norm:
                        # Merge rows
                        merged_row = {**eye_row.to_dict(), **umr_row.to_dict()}
                        merged_rows.append(merged_row)
                        break  # Assume one match per word
                        
            df_merged = pd.DataFrame(merged_rows)
            df_merged.reset_index(drop=True, inplace=True)
            if df_eye is self.df_eye_word_participants:
                df_participants_merged = df_merged
            else:
                df_avg_merged = df_merged
                
        self.df_merged_word_participants = df_participants_merged
        self.df_merged_word_avg = df_avg_merged
        return df_participants_merged, df_avg_merged
    
    
    def run_mixed_effects_word_level_batch(self, umr_features, eye_metrics, max_features_per_model=2, save_path="results/mixed_effects_word_level_results.csv"):
        """
        Run mixed-effects models for multiple UMR feature combinations and eye metrics at word level.
        
        Models:
            eye_metric ~ UMR_features + (1 | subject)
        """

        if self.df_merged_word_participants is None:
            raise ValueError("Word-level participant-level merged data is required.")

        results = []

        # Generate feature combinations
        feature_combinations = []
        for k in range(1, max_features_per_model + 1):
            feature_combinations.extend(itertools.combinations(umr_features, k))

        for eye_metric in eye_metrics:
            for feat_combo in feature_combinations:
                cols = ["subject", eye_metric] + list(feat_combo)
                df_model = self.df_merged_word_participants[cols].dropna()

                # Skip too small datasets
                if df_model["subject"].nunique() < 2 or len(df_model) < 50:
                    continue

                fixed_formula = " + ".join(feat_combo)
                formula = f"{eye_metric} ~ {fixed_formula}"

                try:
                    model = smf.mixedlm(formula, df_model, groups=df_model["subject"])
                    result = model.fit(reml=True, disp=False)
                except Exception as e:
                    print(f"Model failed: {formula}")
                    continue

                # Save fixed effects
                for feat in feat_combo:
                    results.append({
                        "eye_metric": eye_metric,
                        "umr_feature": feat,
                        "feature_set": "+".join(feat_combo),
                        "coef": round(result.params.get(feat, 0), 4),
                        "p_value": round(result.pvalues.get(feat, 1), 6),
                        "intercept": round(result.params.get("Intercept", 0), 4),
                        "random_effect_var": round(result.cov_re.iloc[0, 0], 4),
                        "n_obs": len(df_model),
                        "n_subjects": df_model["subject"].nunique()
                    })

        df_results = pd.DataFrame(results)
        df_results.to_csv(save_path, index=False)
        print(f"Mixed-effects word-level results saved to {save_path}")

        return df_results
    