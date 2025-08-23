from load_data import load_and_clean_data
from analysis import perform_analysis
from insights import generate_insights
from new_insights import generate_additional_insights


def main():
    df = load_and_clean_data("../data/50_Startups.csv")
    perform_analysis(df)
    generate_insights(df)
    generate_additional_insights(df)
    print("All graphs and insights saved to outputs folder!")

if __name__ == "__main__":
    main()
