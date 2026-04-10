import pandas as pd
import sqlite3

# Extract
df = pd.read_csv("./data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Transform
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# Load
conn = sqlite3.connect("churn.db")
df.to_sql("customers", conn, if_exists="replace", index=False)

# Query
query = """
SELECT Contract, COUNT(*) as total_customers
FROM customers
GROUP BY Contract
"""

result = pd.read_sql(query, conn)
print(result)