"""
Module: clean.py
Purpose: Filter rows from the raw CSV based on a specified date range and save the cleaned dataset.
"""

import pandas as pd

# --- Configuration Constants ---
INPUT_FILE = "data/raw/All_external.csv"
OUTPUT_FILE = "data/processed/financial_news_cleaned_2000_2020.csv"
CHUNKSIZE = 500_000
START_DATE = pd.Timestamp("2000-01-01")
END_DATE = pd.Timestamp("2023-12-31")


def filter_by_date() -> None:
    """Filters rows by date range and writes the output CSV."""
    print("Filtering rows by date range...")
    with pd.read_csv(INPUT_FILE, chunksize=CHUNKSIZE) as reader, \
         open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as outfile:
        header_written = False
        total_rows = 0
        kept_rows = 0

        for chunk in reader:
            # Parse and remove timezone awareness
            chunk["Date"] = pd.to_datetime(chunk["Date"], errors="coerce", utc=True).dt.tz_localize(None)
            # Filter within desired range
            filtered = chunk[(chunk["Date"] >= START_DATE) & (chunk["Date"] <= END_DATE)]

            if not header_written:
                filtered.to_csv(outfile, index=False)
                header_written = True
            else:
                filtered.to_csv(outfile, index=False, header=False)

            total_rows += len(chunk)
            kept_rows += len(filtered)
            print(f"Processed chunk — kept {len(filtered)} of {len(chunk)} rows")

    print(f"Done! Final dataset has {kept_rows:,} of {total_rows:,} total rows.")
    print(f"Saved as '{OUTPUT_FILE}'")


def main() -> None:
    filter_by_date()


if __name__ == "__main__":
    main()





