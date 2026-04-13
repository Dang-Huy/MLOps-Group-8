"""
src/data/schema.py
==================
Single source of truth for the raw data contract.

Input  : defined statically by the team from EDA finding
Output : schema dict + column-group lists consumed by validation.py
         and preprocessing.py
"""

# ---------------------------------------------------------------------------
# Expected dtypes after ingestion (before any cleaning)
# ---------------------------------------------------------------------------
RAW_SCHEMA: dict[str, str] = {
    "ID":                       "object",
    "Customer_ID":              "object",
    "Month":                    "object",
    "Name":                     "object",
    "Age":                      "object",   # dirty — contains trailing chars
    "SSN":                      "object",
    "Occupation":               "object",
    "Annual_Income":            "object",   # dirty — trailing underscores
    "Monthly_Inhand_Salary":    "float64",
    "Num_Bank_Accounts":        "int64",
    "Num_Credit_Card":          "int64",
    "Interest_Rate":            "int64",
    "Num_of_Loan":              "object",   # dirty — trailing chars
    "Type_of_Loan":             "object",
    "Delay_from_due_date":      "int64",
    "Num_of_Delayed_Payment":   "object",   # dirty — trailing chars
    "Changed_Credit_Limit":     "object",   # dirty — placeholder '_'
    "Num_Credit_Inquiries":     "float64",
    "Credit_Mix":               "object",
    "Outstanding_Debt":         "object",   # dirty — trailing underscores
    "Credit_Utilization_Ratio": "float64",
    "Credit_History_Age":       "object",   # string like "22 Years and 9 Months"
    "Payment_of_Min_Amount":    "object",
    "Total_EMI_per_month":      "float64",
    "Amount_invested_monthly":  "object",   # dirty — garbage placeholder
    "Payment_Behaviour":        "object",
    "Monthly_Balance":          "object",   # dirty — garbage placeholder
    "Credit_Score":             "object",   # target — absent in test set
}

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------

# Columns dropped before modelling (PII / identifiers)
PII_COLUMNS: list[str] = ["ID", "Customer_ID", "Name", "SSN", "Month"]

# Columns with known garbage/placeholder values requiring cleaning
DIRTY_NUMERIC: list[str] = [
    "Age",
    "Annual_Income",
    "Num_of_Loan",
    "Num_of_Delayed_Payment",
    "Outstanding_Debt",
    "Changed_Credit_Limit",
    "Amount_invested_monthly",
    "Monthly_Balance",
]

DIRTY_CATEGORICAL: list[str] = [
    "Occupation",           # placeholder '_______'
    "Credit_Mix",           # placeholder '_'
    "Payment_Behaviour",    # garbage '!@9#%8'
    "Payment_of_Min_Amount",# unknown 'NM'
]

# Columns with out-of-range values needing capping
OUTLIER_COLUMNS: dict[str, tuple] = {
    "Num_Bank_Accounts":    (0,  20),
    "Num_Credit_Card":      (0,  20),
    "Interest_Rate":        (1, 100),
    "Num_Credit_Inquiries": (0,  20),
    "Num_of_Loan":          (0,  10),
    "Delay_from_due_date":  (0,  None),    # None = no upper cap
    "Num_of_Delayed_Payment": (0, None),
}

# Age valid range
AGE_RANGE: tuple[int, int] = (18, 100)

# Target column
TARGET_COL: str = "Credit_Score"
TARGET_CLASSES: list[str] = ["Poor", "Standard", "Good"]
TARGET_ENCODING: dict[str, int] = {"Poor": 0, "Standard": 1, "Good": 2}

# Columns to drop (in addition to PII) — multi-label free text, too sparse
DROP_BEFORE_MODEL: list[str] = ["Type_of_Loan"]