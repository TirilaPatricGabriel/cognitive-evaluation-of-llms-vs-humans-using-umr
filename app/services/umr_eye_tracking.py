import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import statsmodels.formula.api as smf

class UMREyeTrackingComparator:
    """
    Compare eye-tracking data with UMR data at sentence-level.
    """

    def __init__(self, df_umr_sent: pd.DataFrame, df_eye_sent_avg: pd.DataFrame, df_eye_sent_participants: pd.DataFrame = None):
        self.df_umr_sent = df_umr_sent
        self.df_eye_sent_avg = df_eye_sent_avg
        self.df_eye_sent_participants = df_eye_sent_participants

        # Merge average data
        self.df_avg_merged = pd.merge( df_eye_sent_avg, df_umr_sent, on="sentence_id", how="inner")

        # Merge participant data
        if df_eye_sent_participants is not None:
            self.df_participant_merged = pd.merge(df_eye_sent_participants, df_umr_sent, on="sentence_id", how="inner")
        else:
            self.df_participant_merged = None


    def compute_correlations(self, umr_features=None, eye_metrics=None, return_heatmap=False, participant_level=False):
        if umr_features is None:
            umr_features = [
                "num_nodes", "num_edges", "max_depth", "avg_depth",
                "num_predicates", "num_entities", "predicate_entity_ratio",
                "num_reentrancies", "avg_degree", "max_degree",
                "num_coordination", "num_temporal_quantities"
            ]

        if eye_metrics is None:
            eye_metrics = ["FFD_avg", "GD_avg", "GPT_avg", "TRT_avg", "TRT_sum", "nFix_avg", "nFix_sum", "reading_order_avg"]

        # Choose dataframe
        df_to_use = self.df_participant_merged if participant_level else self.df_avg_merged
        if df_to_use is None:
            raise ValueError("Participant-level data not available.")

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
        corr_matrix = df_to_use[umr_features + eye_metrics].corr().loc[umr_features, eye_metrics]
        corr_matrix = corr_matrix.fillna(0).replace([np.inf, -np.inf], 0)
        corr_matrix_json = corr_matrix.round(2).to_dict()

        # Heatmap
        heatmap_base64 = None
        if return_heatmap:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
            plt.title("Correlation: UMR features vs Eye-tracking metrics")
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            heatmap_base64 = base64.b64encode(buf.read()).decode("utf-8")
            with open("plots/umr_eye_corr_heatmap.png", "wb") as f:
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
            # Compute mean ± SD per UMR x eye pair
            participant_corrs_summary = df_all.groupby(['UMR_feature', 'Eye_metric'])['Correlation'].agg(['mean', 'std']).reset_index()
            participant_corrs_summary['mean'] = participant_corrs_summary['mean'].round(2)
            participant_corrs_summary['std'] = participant_corrs_summary['std'].round(2)
            # Format as mean ± SD
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

        # Plot mean ± SD using errorbar
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
    
    
    def mixed_effects_model(self, umr_features, eye_metric):
        """
        Run a linear mixed-effects model: eye_metric ~ umr_features + (1|subject)
        
        eye_metric = the eye-tracking metric to predict
        umr_features = one or more UMR features as fixed effects
        """

        if self.df_participant_merged is None:
            raise ValueError("Participant-level data is required for mixed-effects modeling.")
        
        # Ensure umr_features is a list
        if isinstance(umr_features, str):
            umr_features = [umr_features]
        
        # Check that all columns exist
        missing_cols = [feat for feat in umr_features if feat not in self.df_participant_merged.columns]
        if eye_metric not in self.df_participant_merged.columns:
            missing_cols.append(eye_metric)
        if missing_cols:
            raise ValueError(f"Columns not found in participant-level data: {missing_cols}")
        
        cols_to_use = ['subject', eye_metric] + umr_features
        df_model = self.df_participant_merged[cols_to_use].dropna()
        
        # Build formula
        fixed_effects_formula = " + ".join(umr_features)
        formula = f"{eye_metric} ~ {fixed_effects_formula}"
        
        # Fit mixed-effects model with random intercept for subject
        model = smf.mixedlm(formula, df_model, groups=df_model["subject"])
        result = model.fit()
        
        # Collect fixed effects results
        fixed_effects = {}
        for feat in umr_features:
            fixed_effects[feat] = (result.params[feat], result.pvalues[feat])
        
        summary_dict = {
            "fixed_effects": fixed_effects,
            "intercept": result.params["Intercept"],
            "random_effect_var": result.cov_re.iloc[0, 0],
            "model_summary": result.summary().as_text()
        }
        
        return summary_dict
