import os
import matplotlib.pyplot as plt
import seaborn as sns

def generate_additional_insights(df):
    os.makedirs("../outputs", exist_ok=True)

    # Calculate ROI and Total Spend if not present
    if "roi" not in df.columns:
        df["total_spend"] = df["r&d_spend"] + df["administration"] + df["marketing_spend"]
        df["roi"] = df["profit"] / df["total_spend"]

    # 1. Average ROI by State
    plt.figure(figsize=(10, 6))
    avg_roi = df.groupby("state")["roi"].mean().sort_values(ascending=False)
    bars = avg_roi.plot(kind="bar", color="mediumseagreen", edgecolor="black")
    plt.title("Average ROI by State", fontsize=14)
    plt.xlabel("State", fontsize=12)
    plt.ylabel("Average ROI", fontsize=12)
    for i, val in enumerate(avg_roi):
        bars.annotate(f"{val:.2f}", (i, val), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig("../outputs/avg_roi_by_state.png")
    plt.close()

    # 2.Profit vs R&D Spend Scatter Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="r&d_spend", y="profit", hue="state", palette="tab10", s=80)
    plt.title("Profit vs R&D Spend", fontsize=14)
    plt.xlabel("R&D Spend")
    plt.ylabel("Profit")
    plt.legend(title="State", fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("../outputs/profit_vs_rd_spend.png")
    plt.close()

    # 3. ROI vs Total Spend Scatter Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="total_spend", y="roi", hue="state", palette="husl", s=80)
    plt.title("ROI vs Total Spend", fontsize=14)
    plt.xlabel("Total Spend")
    plt.ylabel("ROI")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("../outputs/roi_vs_total_spend.png")
    plt.close()