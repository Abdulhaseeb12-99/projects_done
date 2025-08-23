import os
import seaborn as sns
import matplotlib.pyplot as plt

def perform_analysis(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["total_spend"] = df["r&d_spend"] + df["administration"] + df["marketing_spend"]
    df["roi"] = df["profit"] / df["total_spend"]

    os.makedirs("../outputs", exist_ok=True)

    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="crest")
    plt.title("Correlation Matrix (Spend vs Profit)")
    plt.tight_layout()
    plt.savefig("../outputs/correlation_matrix.png")
    plt.close()

    # ROI Distribution with Outliers Highlighted
    plt.figure(figsize=(10, 6))
    sns.histplot(df["roi"], bins=12, kde=True, color="skyblue", edgecolor="black", alpha=0.7)

    # Compute IQR for ROI
    Q1 = df["roi"].quantile(0.25)
    Q3 = df["roi"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Highlighting outliers
    outliers = df[(df["roi"] < lower_bound) | (df["roi"] > upper_bound)]
    mean_roi = df["roi"].mean()
    median_roi = df["roi"].median()
    plt.axvline(mean_roi, color='red', linestyle='--', linewidth=2, label=f"Mean ROI: {mean_roi:.2f}")
    plt.axvline(median_roi, color='green', linestyle='-.', linewidth=2, label=f"Median ROI: {median_roi:.2f}")
    if not outliers.empty:
        first_outlier = outliers["roi"].iloc[0]
        plt.axvline(first_outlier, color='purple', linestyle=':', linewidth=1.5, alpha=0.7, label="Outliers")
        for outlier in outliers["roi"].iloc[1:]:
            plt.axvline(outlier, color='purple', linestyle=':', linewidth=1.5, alpha=0.7)

    plt.title("Distribution of ROI with Outliers Highlighted", fontsize=14)
    plt.xlabel("ROI")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    sns.despine()
    plt.tight_layout()
    plt.savefig("../outputs/roi_distribution.png")
    plt.close()