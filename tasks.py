"""
tasks.py — Task definitions and embedded datasets for DataQualityEnv.

Three tasks with increasing difficulty:
  task1_easy   → Customer records  (deduplication + imputation)
  task2_medium → Sales data        (format standardisation + cleaning)
  task3_hard   → Healthcare records (full pipeline: all issue types)
"""

from typing import Any, Dict

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — Customer Records (Easy)
# Issues: 3 duplicate rows, 5 missing ages, 2 missing emails
# ─────────────────────────────────────────────────────────────────────────────

TASK1_DATA = [
    {"id": 1,  "name": "Alice Johnson",  "email": "alice@mail.com",  "age": 28.0, "city": "Mumbai"},
    {"id": 2,  "name": "Bob Smith",      "email": "bob@mail.com",    "age": 34.0, "city": "Delhi"},
    {"id": 3,  "name": "Carol White",    "email": "carol@mail.com",  "age": None, "city": "Bangalore"},
    {"id": 4,  "name": "David Brown",    "email": None,              "age": 45.0, "city": "Chennai"},
    {"id": 5,  "name": "Eva Green",      "email": "eva@mail.com",    "age": None, "city": "Pune"},
    {"id": 6,  "name": "Frank Lee",      "email": "frank@mail.com",  "age": 31.0, "city": "Hyderabad"},
    {"id": 7,  "name": "Grace Kim",      "email": "grace@mail.com",  "age": None, "city": "Kolkata"},
    # --- duplicates of rows 1, 2, 3 ---
    {"id": 8,  "name": "Alice Johnson",  "email": "alice@mail.com",  "age": 28.0, "city": "Mumbai"},
    {"id": 9,  "name": "Bob Smith",      "email": "bob@mail.com",    "age": 34.0, "city": "Delhi"},
    {"id": 10, "name": "Henry Park",     "email": None,              "age": 52.0, "city": "Jaipur"},
    {"id": 11, "name": "Iris Wang",      "email": "iris@mail.com",   "age": None, "city": "Ahmedabad"},
    {"id": 12, "name": "Carol White",    "email": "carol@mail.com",  "age": None, "city": "Bangalore"},
    {"id": 13, "name": "Jack Davis",     "email": "jack@mail.com",   "age": 38.0, "city": "Surat"},
    {"id": 14, "name": "Kelly Adams",    "email": "kelly@mail.com",  "age": None, "city": "Lucknow"},
    {"id": 15, "name": "Leo Martinez",   "email": "leo@mail.com",    "age": 27.0, "city": "Kanpur"},
]

TASK1_CONFIG: Dict[str, Any] = {
    "name": "Customer Records Deduplication & Imputation",
    "description": (
        "A CRM export contains 15 customer records with duplicate entries and missing fields. "
        "Remove duplicate rows (identified by name + email + city) and fill in missing age and "
        "email values so the dataset is clean and ready for downstream processing."
    ),
    "difficulty": "easy",
    "max_steps": 10,
    "data": TASK1_DATA,
    "schema": {
        "id": "int", "name": "str", "email": "str", "age": "float", "city": "str"
    },
    "dup_subset": ["name", "email", "city"],
    "date_columns": [],
    "phone_columns": [],
    "positive_columns": [],
    "outlier_ranges": {},
    # Scoring weights (must sum to 1.0)
    "score_weights": {
        "duplicate_score": 0.40,
        "age_missing_score": 0.30,
        "email_missing_score": 0.30,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — Sales Data (Medium)
# Issues: 4 bad date formats, 4 bad phone formats, 3 negative amounts, 3 missing regions
# ─────────────────────────────────────────────────────────────────────────────

TASK2_DATA = [
    {"order_id": "ORD001", "date": "2024-01-15",     "amount": 1500.00, "phone": "+91-98765-43210", "region": "North", "product": "Laptop"},
    {"order_id": "ORD002", "date": "15/02/2024",     "amount":  850.00, "phone": "9876543211",      "region": "South", "product": "Mouse"},
    {"order_id": "ORD003", "date": "2024-03-20",     "amount": -200.00, "phone": "+91-87654-32109", "region": None,    "product": "Keyboard"},
    {"order_id": "ORD004", "date": "April 5, 2024",  "amount": 2300.00, "phone": "087654-32108",    "region": "East",  "product": "Monitor"},
    {"order_id": "ORD005", "date": "2024-01-08",     "amount":  450.00, "phone": "+91-76543-21098", "region": "West",  "product": "Headphones"},
    {"order_id": "ORD006", "date": "22-06-2024",     "amount": 3200.00, "phone": "7654321097",      "region": None,    "product": "Tablet"},
    {"order_id": "ORD007", "date": "2024-07-11",     "amount": -150.00, "phone": "+91-65432-10987", "region": "North", "product": "Cable"},
    {"order_id": "ORD008", "date": "08/30/2024",     "amount": 1800.00, "phone": "+91-54321-09876", "region": "South", "product": "Webcam"},
    {"order_id": "ORD009", "date": "2024-09-14",     "amount":  950.00, "phone": "054321-09875",    "region": "East",  "product": "Speaker"},
    {"order_id": "ORD010", "date": "2024-10-03",     "amount": -500.00, "phone": "+91-43210-98764", "region": None,    "product": "Printer"},
    {"order_id": "ORD011", "date": "2024-11-19",     "amount": 1100.00, "phone": "+91-32109-87653", "region": "West",  "product": "Router"},
    {"order_id": "ORD012", "date": "Dec 7, 2024",    "amount":  670.00, "phone": "+91-21098-76542", "region": "North", "product": "Hub"},
    {"order_id": "ORD013", "date": "2024-12-22",     "amount": 4200.00, "phone": "+91-10987-65431", "region": "South", "product": "SSD"},
    {"order_id": "ORD014", "date": "2025-01-05",     "amount":  320.00, "phone": "+91-09876-54320", "region": "East",  "product": "Cable"},
    {"order_id": "ORD015", "date": "2025-02-14",     "amount": 2750.00, "phone": "+91-98765-43219", "region": "West",  "product": "GPU"},
]

TASK2_CONFIG: Dict[str, Any] = {
    "name": "Sales Data Standardisation & Cleansing",
    "description": (
        "An e-commerce export contains 15 sales records with mixed date formats, malformed Indian phone "
        "numbers, data-entry errors (negative revenue), and missing region labels. "
        "Standardise all dates to ISO 8601 (YYYY-MM-DD), normalise phones to +91-XXXXX-XXXXX, "
        "remove negative transactions, and fill missing region values."
    ),
    "difficulty": "medium",
    "max_steps": 15,
    "data": TASK2_DATA,
    "schema": {
        "order_id": "str", "date": "str", "amount": "float",
        "phone": "str", "region": "str", "product": "str"
    },
    "dup_subset": None,
    "date_columns": ["date"],
    "phone_columns": ["phone"],
    "positive_columns": ["amount"],
    "outlier_ranges": {},
    "score_weights": {
        "date_format_score": 0.25,
        "phone_format_score": 0.25,
        "negative_amount_score": 0.25,
        "region_missing_score": 0.25,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — Healthcare Records (Hard)
# Issues: 2 duplicate patient_ids, 6 missing values across key columns,
#         3 date format inconsistencies (dob + last_visit),
#         3 physiologically impossible vitals (outliers),
#         3 negative glucose values (impossible)
# ─────────────────────────────────────────────────────────────────────────────

TASK3_DATA = [
    {"patient_id": "P001", "name": "Arjun Mehta",   "dob": "1985-06-12",  "gender": "M", "bp_systolic": 120, "bp_diastolic": 80,  "glucose": 95.0,   "diagnosis": "Hypertension", "medication": "Amlodipine", "last_visit": "2024-01-10"},
    {"patient_id": "P002", "name": "Priya Sharma",  "dob": "12/03/1992",  "gender": "F", "bp_systolic": 115, "bp_diastolic": 75,  "glucose": 88.0,   "diagnosis": "Diabetes",     "medication": "Metformin",  "last_visit": "2024-02-14"},
    {"patient_id": "P003", "name": "Rohan Gupta",   "dob": "1978-11-25",  "gender": "M", "bp_systolic": 500, "bp_diastolic": 90,  "glucose": 102.0,  "diagnosis": "Asthma",       "medication": "Salbutamol", "last_visit": "March 5, 2024"},
    {"patient_id": "P004", "name": "Divya Nair",    "dob": "1995-04-08",  "gender": "F", "bp_systolic": 118, "bp_diastolic": 78,  "glucose": 5000.0, "diagnosis": None,           "medication": "Metoprolol", "last_visit": "2024-04-02"},
    {"patient_id": "P005", "name": "Karan Singh",   "dob": "07-09-1988",  "gender": "M", "bp_systolic": 130, "bp_diastolic": 85,  "glucose": 110.0,  "diagnosis": "Hypertension", "medication": None,         "last_visit": "2024-05-19"},
    {"patient_id": "P006", "name": "Sneha Reddy",   "dob": "1990-02-17",  "gender": "F", "bp_systolic": 112, "bp_diastolic": 70,  "glucose": 92.0,   "diagnosis": "Thyroid",      "medication": "Levothyrox", "last_visit": "2024-06-11"},
    {"patient_id": "P007", "name": "Vikram Iyer",   "dob": "1982-08-30",  "gender": "M", "bp_systolic": 140, "bp_diastolic": 300, "glucose": 98.0,   "diagnosis": "Diabetes",     "medication": "Insulin",    "last_visit": "2024-07-08"},
    {"patient_id": "P008", "name": "Anita Joshi",   "dob": "1975-12-05",  "gender": "F", "bp_systolic": 125, "bp_diastolic": 82,  "glucose": -45.0,  "diagnosis": "Hypertension", "medication": "Atenolol",   "last_visit": "2024-08-22"},
    {"patient_id": "P009", "name": "Suresh Pillai", "dob": "June 20 1993","gender": "M", "bp_systolic": 118, "bp_diastolic": 76,  "glucose": 105.0,  "diagnosis": None,           "medication": "Aspirin",    "last_visit": "2024-09-15"},
    {"patient_id": "P010", "name": "Meena Patel",   "dob": "1987-03-14",  "gender": "F", "bp_systolic": 110, "bp_diastolic": 68,  "glucose": 87.0,   "diagnosis": "Anaemia",      "medication": "Iron",       "last_visit": "2024-10-03"},
    # Duplicates of P001 and P003
    {"patient_id": "P001", "name": "Arjun Mehta",   "dob": "1985-06-12",  "gender": "M", "bp_systolic": 120, "bp_diastolic": 80,  "glucose": 95.0,   "diagnosis": "Hypertension", "medication": "Amlodipine", "last_visit": "2024-01-10"},
    {"patient_id": "P011", "name": "Rahul Verma",   "dob": "1991-07-22",  "gender": "M", "bp_systolic": 122, "bp_diastolic": 79,  "glucose": None,   "diagnosis": "Migraine",     "medication": "Sumatriptan","last_visit": "2024-11-17"},
    {"patient_id": "P012", "name": "Lakshmi Das",   "dob": "1969-09-01",  "gender": "F", "bp_systolic": 145, "bp_diastolic": 92,  "glucose": 130.0,  "diagnosis": "Diabetes",     "medication": "Glipizide",  "last_visit": "2024-12-05"},
    {"patient_id": "P013", "name": "Aditya Roy",    "dob": "2000-01-15",  "gender": "M", "bp_systolic": 108, "bp_diastolic": 65,  "glucose": 80.0,   "diagnosis": None,           "medication": None,         "last_visit": "2025-01-09"},
    {"patient_id": "P003", "name": "Rohan Gupta",   "dob": "1978-11-25",  "gender": "M", "bp_systolic": 500, "bp_diastolic": 90,  "glucose": 102.0,  "diagnosis": "Asthma",       "medication": "Salbutamol", "last_visit": "March 5, 2024"},
    {"patient_id": "P014", "name": "Pooja Mishra",  "dob": "1996-05-28",  "gender": "F", "bp_systolic": 116, "bp_diastolic": 74,  "glucose": 91.0,   "diagnosis": "PCOS",         "medication": "Metformin",  "last_visit": "2025-02-18"},
    {"patient_id": "P015", "name": "Nitin Kumar",   "dob": "1983-10-09",  "gender": "M", "bp_systolic": 135, "bp_diastolic": 88,  "glucose": -20.0,  "diagnosis": "Hypertension", "medication": "Losartan",   "last_visit": "2025-03-01"},
    {"patient_id": "P016", "name": "Swati Ghosh",   "dob": "1979-02-14",  "gender": "F", "bp_systolic": 128, "bp_diastolic": 84,  "glucose": 99.0,   "diagnosis": "Thyroid",      "medication": None,         "last_visit": "2025-03-12"},
    {"patient_id": "P017", "name": "Rajesh Bansal", "dob": "1965-12-30",  "gender": "M", "bp_systolic": 160, "bp_diastolic": 100, "glucose": 145.0,  "diagnosis": "Diabetes",     "medication": "Insulin",    "last_visit": "2025-03-20"},
    {"patient_id": "P018", "name": "Kavita Rao",    "dob": "1988-08-05",  "gender": "F", "bp_systolic": 119, "bp_diastolic": 77,  "glucose": 94.0,   "diagnosis": "Anaemia",      "medication": "Iron",       "last_visit": "2025-03-25"},
]

TASK3_CONFIG: Dict[str, Any] = {
    "name": "Healthcare Records Full Quality Pipeline",
    "description": (
        "A hospital export of 20 patient records contains a range of quality issues: duplicate patient IDs, "
        "missing diagnoses and medications, inconsistent date formats in dob and last_visit fields, "
        "physiologically impossible vital signs (bp_systolic=500, bp_diastolic=300, negative glucose), "
        "and other anomalies. Apply the full data-quality pipeline to bring all records to clinical standard."
    ),
    "difficulty": "hard",
    "max_steps": 25,
    "data": TASK3_DATA,
    "schema": {
        "patient_id": "str", "name": "str", "dob": "str", "gender": "str",
        "bp_systolic": "int", "bp_diastolic": "int", "glucose": "float",
        "diagnosis": "str", "medication": "str", "last_visit": "str",
    },
    "dup_subset": ["patient_id"],
    "date_columns": ["dob", "last_visit"],
    "phone_columns": [],
    "positive_columns": ["glucose"],
    "outlier_ranges": {
        "bp_systolic":  (60, 200),
        "bp_diastolic": (40, 130),
        "glucose":      (50, 500),
    },
    "score_weights": {
        "duplicate_score":      0.20,
        "missing_value_score":  0.20,
        "date_format_score":    0.20,
        "outlier_score":        0.20,
        "negative_vital_score": 0.20,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Master registry
# ─────────────────────────────────────────────────────────────────────────────

TASKS: Dict[str, Dict] = {
    "task1_easy":   TASK1_CONFIG,
    "task2_medium": TASK2_CONFIG,
    "task3_hard":   TASK3_CONFIG,
}
