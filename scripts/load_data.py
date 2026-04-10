import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"))

files = {
    "admissions": "data/raw/ADMISSIONS.csv",
    "patients": "data/raw/PATIENTS.csv",
    "diagnoses_icd": "data/raw/DIAGNOSES_ICD.csv",
    "prescriptions": "data/raw/PRESCRIPTIONS.csv",
    "noteevents": "data/raw/NOTEEVENTS.csv",
}

for table, path in files.items():
    print(f"Loading {table}...")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower()
    df.to_sql(table, engine, if_exists="replace", index=False)
    print(f"  {len(df)} rows loaded into '{table}'")

print("Done.")
