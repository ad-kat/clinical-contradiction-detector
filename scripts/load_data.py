import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"))

files = {
    "admissions": "data/raw/admissions.csv.gz",
    "patients": "data/raw/patients.csv.gz",
    "diagnoses_icd": "data/raw/diagnoses_icd.csv.gz",
    "prescriptions": "data/raw/prescriptions.csv.gz",
    "discharge_notes": "data/raw/discharge.csv.gz",
}

for table, path in files.items():
    print(f"Loading {table}...")
    df = pd.read_csv(path, low_memory=False, compression="gzip")
    df.columns = df.columns.str.lower()
    df.to_sql(table, engine, if_exists="replace", index=False, chunksize=10000)
    print(f"  {len(df)} rows loaded into '{table}'")

print("Done.")