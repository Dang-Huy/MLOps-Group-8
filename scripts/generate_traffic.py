"""Generate synthetic prediction traffic for monitoring validation."""
import argparse
import time
import random
import json
import sys
try:
    import httpx
except ImportError:
    import urllib.request as _req
    httpx = None


PROFILES = [
    # Likely Good
    {"Age": 35, "Annual_Income": 85000, "Monthly_Inhand_Salary": 6500,
     "Num_Bank_Accounts": 3, "Num_Credit_Card": 4, "Interest_Rate": 7,
     "Num_of_Loan": 2, "Delay_from_due_date": 0, "Num_of_Delayed_Payment": 0,
     "Changed_Credit_Limit": 5.0, "Num_Credit_Inquiries": 1,
     "Credit_Mix": "Good", "Outstanding_Debt": 8000,
     "Credit_Utilization_Ratio": 18.0, "Credit_History_Age": "8 Years and 4 Months",
     "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 350,
     "Amount_invested_monthly": 800, "Payment_Behaviour": "High_spent_Large_value_payments",
     "Monthly_Balance": 3200, "Occupation": "Engineer"},
    # Likely Standard
    {"Age": 28, "Annual_Income": 42000, "Monthly_Inhand_Salary": 3000,
     "Num_Bank_Accounts": 4, "Num_Credit_Card": 5, "Interest_Rate": 14,
     "Num_of_Loan": 3, "Delay_from_due_date": 8, "Num_of_Delayed_Payment": 3,
     "Changed_Credit_Limit": 2.0, "Num_Credit_Inquiries": 5,
     "Credit_Mix": "Standard", "Outstanding_Debt": 15000,
     "Credit_Utilization_Ratio": 42.0, "Credit_History_Age": "3 Years and 6 Months",
     "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 650,
     "Amount_invested_monthly": 150, "Payment_Behaviour": "Low_spent_Medium_value_payments",
     "Monthly_Balance": 800, "Occupation": "Teacher"},
    # Likely Poor
    {"Age": 22, "Annual_Income": 18000, "Monthly_Inhand_Salary": 1200,
     "Num_Bank_Accounts": 7, "Num_Credit_Card": 9, "Interest_Rate": 22,
     "Num_of_Loan": 7, "Delay_from_due_date": 25, "Num_of_Delayed_Payment": 14,
     "Changed_Credit_Limit": 0.5, "Num_Credit_Inquiries": 12,
     "Credit_Mix": "Bad", "Outstanding_Debt": 28000,
     "Credit_Utilization_Ratio": 85.0, "Credit_History_Age": "1 Years and 2 Months",
     "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 1200,
     "Amount_invested_monthly": 0, "Payment_Behaviour": "High_spent_Small_value_payments",
     "Monthly_Balance": -200, "Occupation": "Scientist"},
    {"Age": 45, "Annual_Income": 120000, "Monthly_Inhand_Salary": 9500,
     "Num_Bank_Accounts": 2, "Num_Credit_Card": 3, "Interest_Rate": 6,
     "Num_of_Loan": 1, "Delay_from_due_date": 0, "Num_of_Delayed_Payment": 0,
     "Changed_Credit_Limit": 8.0, "Num_Credit_Inquiries": 0,
     "Credit_Mix": "Good", "Outstanding_Debt": 5000,
     "Credit_Utilization_Ratio": 12.0, "Credit_History_Age": "15 Years and 0 Months",
     "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 200,
     "Amount_invested_monthly": 2000, "Payment_Behaviour": "High_spent_Large_value_payments",
     "Monthly_Balance": 5000, "Occupation": "Doctor"},
    {"Age": 31, "Annual_Income": 55000, "Monthly_Inhand_Salary": 4000,
     "Num_Bank_Accounts": 3, "Num_Credit_Card": 6, "Interest_Rate": 11,
     "Num_of_Loan": 4, "Delay_from_due_date": 5, "Num_of_Delayed_Payment": 2,
     "Changed_Credit_Limit": 3.0, "Num_Credit_Inquiries": 3,
     "Credit_Mix": "Standard", "Outstanding_Debt": 12000,
     "Credit_Utilization_Ratio": 35.0, "Credit_History_Age": "5 Years and 8 Months",
     "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 500,
     "Amount_invested_monthly": 300, "Payment_Behaviour": "Low_spent_Large_value_payments",
     "Monthly_Balance": 1500, "Occupation": "Lawyer"},
    {"Age": 19, "Annual_Income": 14000, "Monthly_Inhand_Salary": 900,
     "Num_Bank_Accounts": 6, "Num_Credit_Card": 8, "Interest_Rate": 25,
     "Num_of_Loan": 6, "Delay_from_due_date": 30, "Num_of_Delayed_Payment": 18,
     "Changed_Credit_Limit": 0.2, "Num_Credit_Inquiries": 15,
     "Credit_Mix": "Bad", "Outstanding_Debt": 32000,
     "Credit_Utilization_Ratio": 90.0, "Credit_History_Age": "0 Years and 8 Months",
     "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 1500,
     "Amount_invested_monthly": 0, "Payment_Behaviour": "High_spent_Small_value_payments",
     "Monthly_Balance": -500, "Occupation": "Student"},
    {"Age": 52, "Annual_Income": 95000, "Monthly_Inhand_Salary": 7200,
     "Num_Bank_Accounts": 2, "Num_Credit_Card": 2, "Interest_Rate": 8,
     "Num_of_Loan": 1, "Delay_from_due_date": 1, "Num_of_Delayed_Payment": 0,
     "Changed_Credit_Limit": 6.0, "Num_Credit_Inquiries": 1,
     "Credit_Mix": "Good", "Outstanding_Debt": 6000,
     "Credit_Utilization_Ratio": 15.0, "Credit_History_Age": "20 Years and 3 Months",
     "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 280,
     "Amount_invested_monthly": 1500, "Payment_Behaviour": "High_spent_Large_value_payments",
     "Monthly_Balance": 4200, "Occupation": "Manager"},
]


def send_single(host: str, profile: dict) -> bool:
    url = f"{host}/predict"
    data = json.dumps(profile).encode()
    try:
        if httpx:
            r = httpx.post(url, content=data,
                           headers={"Content-Type": "application/json"}, timeout=5)
            return r.status_code == 200
        else:
            req = _req.Request(url, data=data,
                               headers={"Content-Type": "application/json"})
            with _req.urlopen(req, timeout=5):
                return True
    except Exception:
        return False


def send_batch(host: str, profiles: list) -> bool:
    url = f"{host}/predict/batch"
    data = json.dumps({"records": profiles}).encode()
    try:
        if httpx:
            r = httpx.post(url, content=data,
                           headers={"Content-Type": "application/json"}, timeout=10)
            return r.status_code == 200
        else:
            req = _req.Request(url, data=data,
                               headers={"Content-Type": "application/json"})
            with _req.urlopen(req, timeout=10):
                return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate prediction traffic")
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--rps", type=float, default=2.0, help="Requests per second")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    args = parser.parse_args()

    interval = 1.0 / args.rps
    deadline = time.time() + args.duration
    ok = fail = 0

    print(f"Sending traffic to {args.host} at {args.rps} rps for {args.duration}s ...")
    while time.time() < deadline:
        profile = random.choice(PROFILES)
        if random.random() < 0.8:
            success = send_single(args.host, profile)
        else:
            batch = random.sample(PROFILES, k=min(3, len(PROFILES)))
            success = send_batch(args.host, batch)
        if success:
            ok += 1
        else:
            fail += 1
        time.sleep(interval)

    total = ok + fail
    print(f"\nDone. {total} requests: {ok} ok, {fail} failed "
          f"({100*fail/total:.1f}% error rate)")


if __name__ == "__main__":
    main()
