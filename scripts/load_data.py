"""
load_data.py -- loads MIMIC-IV .csv.gz files into postgres

run this once from project root after setting up your .env:
    python scripts/load_data.py

takes a while for prescriptions (20M+ rows), that's normal.
"""

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"))

# map table name -> file path
# discharge notes are in a separate gz file (MIMIC-IV-Note)
files = {
    "admissions":    "data/raw/admissions.csv.gz",
    "patients":      "data/raw/patients.csv.gz",
    "diagnoses_icd": "data/raw/diagnoses_icd.csv.gz",
    "prescriptions": "data/raw/prescriptions.csv.gz",
    "discharge_notes": "data/raw/discharge.csv.gz",
}

for table, path in files.items():
    print(f"Loading {table}...")
    df = pd.read_csv(path, low_memory=False, compression="gzip")
    df.columns = df.columns.str.lower()
    # if_exists=replace so re-running this script doesn't append duplicates
    df.to_sql(table, engine, if_exists="replace", index=False, chunksize=10000)
    print(f"  {len(df)} rows -> '{table}'")

print("Done.")