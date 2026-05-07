Industry Use Case
Banks, payment gateways, UPI systems, credit card companies, and FinTech platforms use AI-based fraud detection systems to identify suspicious financial transactions in real time.

Fraud detection systems help financial institutions:

Prevent financial loss
Detect suspicious behavior
Block unauthorized transactions
Protect customer accounts
Reduce cybercrime risk
Lab Objective
In this lab, students will learn how to:

Create a synthetic transaction dataset.
Understand fraud detection input parameters.
Identify suspicious transaction patterns.
Classify transactions into Fraud and Non-Fraud categories.
Detect anomalies using rule-based AI logic.
Visualize fraud patterns.
Understand how AI supports fraud prevention systems.
Real-World Relevance
This lab simulates fraud detection workflows used in:

Credit card fraud monitoring
UPI fraud detection
Online banking security
ATM transaction monitoring
Insurance fraud analytics
E-commerce payment security
Dataset Note
This dataset is synthetic and created only for educational purposes.


[ ]
# CELL 1: Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

print("Libraries imported successfully.")

[ ]
# CELL 2: Create realistic transaction-related master lists

np.random.seed(42)

customer_names = [
    "Aarav Sharma", "Priya Mehta", "Rohan Verma", "Neha Gupta",
    "Karan Malhotra", "Ananya Singh", "Vikram Rao", "Sneha Iyer",
    "Rahul Nair", "Meera Joshi"
]

cities = [
    "Delhi", "Mumbai", "Bengaluru", "Hyderabad",
    "Pune", "Chennai", "Kolkata", "Noida"
]

transaction_types = [
    "UPI", "Credit Card", "Debit Card", "Net Banking", "ATM Withdrawal"
]

device_types = [
    "Mobile", "Laptop", "Desktop", "Tablet"
]

print("Master lists created successfully.")

[ ]
# CELL 3: Generate synthetic banking transaction dataset

n = 200

df = pd.DataFrame({
    "Transaction_ID": range(5001, 5001 + n),
    "Customer_Name": np.random.choice(customer_names, n),
    "Customer_City": np.random.choice(cities, n),
    "Transaction_Amount": np.random.randint(100, 500000, n),
    "Transaction_Type": np.random.choice(transaction_types, n),
    "Device_Type": np.random.choice(device_types, n),
    "Hour_of_Transaction": np.random.randint(0, 24, n),
    "Transactions_Last_24Hrs": np.random.randint(1, 30, n),
    "International_Transaction": np.random.choice([0, 1], n),
    "Failed_Login_Attempts": np.random.randint(0, 8, n),
    "Account_Age_Years": np.random.randint(0, 15, n)
})

df.head(10)

[ ]
# CELL 4: Create fraud detection logic
# This is a simplified rule-based fraud analytics model.

fraud_probability = []
fraud_flag = []
fraud_reason = []
bank_action = []

for index, row in df.iterrows():

    score = 0
    reasons = []

    # Large transaction amount
    if row["Transaction_Amount"] > 200000:
        score += 0.25
        reasons.append("Very high transaction amount")

    # Late night transaction
    if row["Hour_of_Transaction"] >= 1 and row["Hour_of_Transaction"] <= 4:
        score += 0.15
        reasons.append("Late night transaction")

    # Many transactions in short time
    if row["Transactions_Last_24Hrs"] > 20:
        score += 0.20
        reasons.append("Unusual transaction frequency")

    # International transaction
    if row["International_Transaction"] == 1:
        score += 0.20
        reasons.append("International transaction")

    # Multiple failed login attempts
    if row["Failed_Login_Attempts"] >= 4:
        score += 0.20
        reasons.append("Multiple failed login attempts")

    fraud_probability.append(round(score, 2))

    # Fraud classification
    if score >= 0.60:
        fraud_flag.append("Fraud")
        bank_action.append("Block transaction and trigger investigation")

    elif score >= 0.35:
        fraud_flag.append("Suspicious")
        bank_action.append("Manual verification required")

    else:
        fraud_flag.append("Normal")
        bank_action.append("Allow transaction")

    fraud_reason.append(", ".join(reasons) if reasons else "Normal customer behavior")

df["Fraud_Probability"] = fraud_probability
df["Fraud_Status"] = fraud_flag
df["Fraud_Reason"] = fraud_reason
df["Bank_Action"] = bank_action

df.head(10)

[ ]
# CELL 5: View complete fraud analytics profile

selected_columns = [
    "Transaction_ID", "Customer_Name", "Customer_City",
    "Transaction_Amount", "Transaction_Type", "Device_Type",
    "Hour_of_Transaction", "Transactions_Last_24Hrs",
    "International_Transaction", "Failed_Login_Attempts",
    "Fraud_Probability", "Fraud_Status",
    "Fraud_Reason", "Bank_Action"
]

df[selected_columns].head(10)

[ ]
# CELL 6: Fraud status distribution

fraud_distribution = df["Fraud_Status"].value_counts()

print("Fraud Status Distribution:")
print(fraud_distribution)

[ ]
# CELL 7: Visualize fraud status distribution

plt.figure(figsize=(7,5))
sns.countplot(data=df, x="Fraud_Status", order=["Normal", "Suspicious", "Fraud"])
plt.title("Fraud Detection Distribution")
plt.xlabel("Fraud Status")
plt.ylabel("Number of Transactions")
plt.show()

[ ]
# CELL 8: Analyze transaction amount by fraud status

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Fraud_Status", y="Transaction_Amount",
            order=["Normal", "Suspicious", "Fraud"])
plt.title("Transaction Amount by Fraud Status")
plt.xlabel("Fraud Status")
plt.ylabel("Transaction Amount")
plt.show()

[ ]
# CELL 9: Analyze failed login attempts

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Fraud_Status", y="Failed_Login_Attempts",
            order=["Normal", "Suspicious", "Fraud"])
plt.title("Failed Login Attempts by Fraud Status")
plt.xlabel("Fraud Status")
plt.ylabel("Failed Login Attempts")
plt.show()

[ ]
# CELL 10: Identify highly suspicious transactions

high_risk_transactions = df[df["Fraud_Status"] == "Fraud"]

high_risk_transactions[selected_columns].head(10)

[ ]
# CELL 11: Create transaction explanation function

def explain_transaction(transaction_id):

    transaction = df[df["Transaction_ID"] == transaction_id]

    if transaction.empty:
        print("Transaction ID not found.")
        return

    row = transaction.iloc[0]

    print("Transaction Fraud Analysis")
    print("--------------------------")
    print(f"Customer Name: {row['Customer_Name']}")
    print(f"Transaction Amount: ₹{row['Transaction_Amount']}")
    print(f"Transaction Type: {row['Transaction_Type']}")
    print(f"Hour of Transaction: {row['Hour_of_Transaction']}")
    print(f"Transactions in Last 24 Hours: {row['Transactions_Last_24Hrs']}")
    print(f"International Transaction: {row['International_Transaction']}")
    print(f"Failed Login Attempts: {row['Failed_Login_Attempts']}")
    print(f"Fraud Probability: {row['Fraud_Probability']}")
    print(f"Fraud Status: {row['Fraud_Status']}")
    print(f"Fraud Reason: {row['Fraud_Reason']}")
    print(f"Recommended Bank Action: {row['Bank_Action']}")

# Example
explain_transaction(5001)

[ ]
# CELL 12: Export fraud detection dataset

df.to_csv("synthetic_fraud_detection_dataset.csv", index=False)

print("Fraud detection dataset exported successfully.")

[ ]
# CELL 13: Download dataset in Google Colab

try:
    from google.colab import files
    files.download("synthetic_fraud_detection_dataset.csv")
except ImportError:
    print("Download works only in Google Colab.")
    print("Dataset saved locally.")

[ ]
# CELL 14: Create fraud summary report

summary_report = df.groupby("Fraud_Status").agg(
    Total_Transactions=("Transaction_ID", "count"),
    Average_Transaction_Amount=("Transaction_Amount", "mean"),
    Average_Failed_Logins=("Failed_Login_Attempts", "mean"),
    Average_Fraud_Probability=("Fraud_Probability", "mean")
).round(2)

summary_report.to_csv("fraud_summary_report.csv")

summary_report

[ ]
# CELL 15: Final outcomes and industry relevance

print("LAB OUTCOMES")
print("------------")
print("1. Students created a synthetic banking transaction dataset.")
print("2. Students understood fraud detection input parameters.")
print("3. Students identified suspicious financial behavior.")
print("4. Students classified transactions into Normal, Suspicious, and Fraud categories.")
print("5. Students visualized fraud patterns using analytics.")
print("6. Students learned how AI supports fraud detection systems.")

print("\nREAL-WORLD INDUSTRY RELEVANCE")
print("-----------------------------")
print("Banks, UPI platforms, and FinTech firms use similar fraud detection systems")
print("to monitor transactions in real time, reduce financial loss, and prevent cyber fraud.")
Student Mini Assignment
Students should:

Identify 5 fraudulent transactions and explain why they were flagged.
Identify 5 normal transactions and explain why they are considered safe.
Explain how AI improves fraud detection compared to manual monitoring.
Explain why anomaly detection is important in banking systems.
Suggest one improvement to make the fraud detection model more realistic.
Suggested GitHub Folder Structure
Fraud-Detection-Analytics-Lab/
│
├── README.md
├── notebook/
│   └── Fraud_Detection_Analytics_Lab.ipynb
├── dataset/
│   └── synthetic_fraud_detection_dataset.csv
├── screenshots/
│   ├── fraud_distribution.png
│   ├── transaction_amount_analysis.png
│   └── failed_login_analysis.png
└── report/
    └── fraud_detection_report.pdf
# Fraud-Detection-Analytics-Lab-using-Synthetic-Banking-Transactions
