import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# Load Together API key
load_dotenv()
client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.getenv("TOGETHER_API_KEY")
)

# CSV absolute path (update if needed)
csv_path = r"E:\git\MintyCraft\data\50_Startups.csv"

# Output folder
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
os.makedirs(output_dir, exist_ok=True)

# Generate summary using Together.ai
def generate_together_summary(df_stats: str) -> str:
    prompt = f"""
You are writing a short report for company managers.

Give a simple and clear summary in one paragraph. Do not list any data or numbers.

Talk about:
- Which companies had the best ROI
- Which states made the most profit
- Which companies spent money well

Here is the data:
{df_stats}
"""
    try:
        response = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            messages=[
                {"role": "system", "content": "You are an expert business analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=350
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Together.ai API Error: {e}"

# Evaluate how accurate the summary is
def evaluate_summary_accuracy(summary: str, df: pd.DataFrame) -> float:
    summary = summary.lower()

    df["total_spend"] = df["r&d_spend"] + df["administration"] + df["marketing_spend"]
    df["roi"] = df["profit"] / df["total_spend"]
    top_roi_company = df.loc[df["roi"].idxmax()]
    low_roi_company = df.loc[df["roi"].idxmin()]
    top_state = df.groupby("state")["profit"].mean().idxmax().lower()

    expected = {
        "top_state": top_state,
        "top_roi_state": top_roi_company["state"].lower(),
        "low_roi_state": low_roi_company["state"].lower(),
    }

    matched = 0
    for val in expected.values():
        if re.search(rf"\b{re.escape(val)}\b", summary):
            matched += 1

    accuracy = (matched / len(expected)) * 100
    return round(accuracy, 2)

# Generate insights and plots
def generate_insights(df):
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]

    df["total_spend"] = df["r&d_spend"] + df["administration"] + df["marketing_spend"]
    df["roi"] = df["profit"] / df["total_spend"]

    top_roi = df.sort_values(by="roi", ascending=False).head(5)
    low_roi = df.sort_values(by="roi").head(5)

    combined = pd.concat([
        top_roi.assign(group="Top ROI"),
        low_roi.assign(group="Low ROI")
    ])

    # Plot 1: Total Spend Comparison
    plt.figure(figsize=(12, 7))
    bar = sns.barplot(data=combined, x="group", y="total_spend", hue="state", palette="Set2")
    plt.title("Total Spend Comparison: Top vs Low ROI Companies")
    plt.xlabel("ROI Group")
    plt.ylabel("Total Spend (USD)")
    for p in bar.patches:
        height = p.get_height()
        bar.annotate(f'{height:,.0f}', (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spending_comparison.png"))
    plt.close()

    # Plot 2: Avg Profit by State
    plt.figure(figsize=(10, 6))
    state_avg_profit = df.groupby("state")["profit"].mean().sort_values(ascending=False)
    bars = state_avg_profit.plot(kind="bar", color="steelblue", edgecolor="black")
    plt.title("Average Profit by State")
    plt.xlabel("State")
    plt.ylabel("Average Profit (USD)")
    for i, val in enumerate(state_avg_profit):
        bars.annotate(f'{val:,.0f}', xy=(i, val), xytext=(0, 5), textcoords="offset points",
                      ha='center', va='bottom', fontsize=9, color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "profit_by_state.png"))
    plt.close()

    # Text Summary for Together.ai
    summary_str = f"""
Top 5 ROI Companies:
{top_roi[['state', 'total_spend', 'roi', 'profit']].to_string(index=False)}

Bottom 5 ROI Companies:
{low_roi[['state', 'total_spend', 'roi', 'profit']].to_string(index=False)}

Average Profit by State:
{state_avg_profit.to_string()}
"""

    summary = generate_together_summary(summary_str)
    accuracy = evaluate_summary_accuracy(summary, df)

    with open(os.path.join(output_dir, "data_insights_summary_together.txt"), "w", encoding="utf-8") as f:
        f.write("Together.ai-Generated Summary:\n\n")
        f.write(summary)
        f.write(f"\n\nAccuracy Score: {accuracy}%")

    print("Visuals saved and summary generated using Together.ai!")
    print(f"Summary Accuracy: {accuracy}%")

# Run the script
if __name__ == "__main__":
    try:
        df = pd.read_csv(csv_path)
        print(" CSV file loaded")
        generate_insights(df)
    except Exception as e:
        print(f" Error: {e}")
