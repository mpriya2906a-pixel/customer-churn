import os
import numpy as np
import pandas as pd


def generate_synthetic_churn_data(
    n_samples: int = 7000,
    random_state: int = 42,
    output_path: str = "data/telco_customer_churn_synthetic.csv"
) -> pd.DataFrame:
    """
    Generates a realistic synthetic customer churn dataset following the Telco structure.
    
    Columns:
    - customer_id: Unique customer identifier
    - tenure: Months with the company (0 - 72)
    - monthly_charges: Monthly bill amount in USD (18.0 - 120.0)
    - total_charges: Cumulative charges (with realistic missing values)
    - contract_type: 'Month-to-month', 'One year', 'Two year'
    - payment_method: 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    - internet_service: 'DSL', 'Fiber optic', 'No'
    - support_tickets: Customer service calls/tickets in last 6 months (0 - 8)
    - last_login_days_ago: Recency of portal or app login (0 - 60 days)
    - usage_frequency: Active usage events/sessions per month (1.0 - 100.0)
    - churn_label: Binary target (0: Retained, 1: Churned)
    """
    rng = np.random.default_rng(random_state)
    
    # 1. Customer IDs
    customer_ids = [f"CUST-{i+10001:05d}" for i in range(n_samples)]
    
    # 2. Tenure (months) - mixture of new customers and long-term loyalists
    tenure_raw = np.concatenate([
        rng.integers(1, 12, size=int(n_samples * 0.35)),     # New customers (< 1 yr)
        rng.integers(12, 36, size=int(n_samples * 0.30)),    # Mid-term (1-3 yrs)
        rng.integers(36, 73, size=n_samples - int(n_samples * 0.35) - int(n_samples * 0.30)) # Loyal
    ])
    rng.shuffle(tenure_raw)
    tenure = tenure_raw
    
    # 3. Contract Type
    # Probability depends somewhat on tenure
    contract_choices = ['Month-to-month', 'One year', 'Two year']
    contract_type = []
    for t in tenure:
        if t < 12:
            probs = [0.80, 0.15, 0.05]
        elif t < 36:
            probs = [0.45, 0.35, 0.20]
        else:
            probs = [0.20, 0.35, 0.45]
        contract_type.append(rng.choice(contract_choices, p=probs))
    contract_type = np.array(contract_type)
    
    # 4. Internet Service
    internet_choices = ['DSL', 'Fiber optic', 'No']
    internet_probs = [0.35, 0.45, 0.20]
    internet_service = rng.choice(internet_choices, size=n_samples, p=internet_probs)
    
    # 5. Monthly Charges ($18 - $120) influenced by internet service
    monthly_charges = []
    for service in internet_service:
        if service == 'No':
            charges = rng.normal(loc=22.0, scale=4.0)
        elif service == 'DSL':
            charges = rng.normal(loc=55.0, scale=12.0)
        else: # Fiber optic
            charges = rng.normal(loc=88.0, scale=15.0)
        monthly_charges.append(np.clip(charges, 18.0, 120.0))
    monthly_charges = np.round(np.array(monthly_charges), 2)
    
    # 6. Total Charges = tenure * monthly_charges + small noise
    # Introduce ~1% missing values to reflect real-world dirty data / tenure==0 blanks
    total_charges = (tenure * monthly_charges) + rng.normal(0, 15.0, size=n_samples)
    total_charges = np.maximum(total_charges, 0.0)
    total_charges = np.round(total_charges, 2).astype(object)
    
    # Inject missing values for ~1.5% of rows
    missing_indices = rng.choice(n_samples, size=int(n_samples * 0.015), replace=False)
    total_charges[missing_indices] = np.nan
    
    # 7. Payment Method
    payment_choices = [
        'Electronic check', 
        'Mailed check', 
        'Bank transfer (automatic)', 
        'Credit card (automatic)'
    ]
    payment_method = rng.choice(payment_choices, size=n_samples, p=[0.35, 0.20, 0.23, 0.22])
    
    # 8. Support Tickets (0 - 8 tickets)
    # Higher for fiber optic customers or random service issues
    support_tickets = rng.poisson(lam=1.6, size=n_samples)
    support_tickets = np.clip(support_tickets, 0, 8)
    
    # 9. Last Login Days Ago (0 - 60 days)
    last_login_days_ago = rng.integers(0, 61, size=n_samples)
    
    # 10. Usage Frequency (1.0 to 100.0 sessions/month)
    usage_frequency = np.round(rng.normal(loc=45.0, scale=18.0, size=n_samples), 1)
    usage_frequency = np.clip(usage_frequency, 1.0, 100.0)
    
    # 11. Churn Label (Realistic probabilistic modeling based on known churn drivers)
    # Latent churn risk score z
    z = (
        -1.40
        + 1.70 * (contract_type == 'Month-to-month')
        - 1.10 * (contract_type == 'Two year')
        - 0.045 * tenure
        + 0.018 * (monthly_charges - 60.0)
        + 0.50 * (internet_service == 'Fiber optic')
        + 0.40 * (payment_method == 'Electronic check')
        + 0.35 * support_tickets
        + 0.035 * (last_login_days_ago - 20.0)
        - 0.025 * (usage_frequency - 40.0)
        + rng.normal(0, 0.45, size=n_samples) # Unobserved variation / noise
    )
    
    # Logistic sigmoid
    prob_churn = 1.0 / (1.0 + np.exp(-z))
    churn_label = (rng.uniform(0, 1, size=n_samples) < prob_churn).astype(int)
    
    # Construct DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract_type': contract_type,
        'payment_method': payment_method,
        'internet_service': internet_service,
        'support_tickets': support_tickets,
        'last_login_days_ago': last_login_days_ago,
        'usage_frequency': usage_frequency,
        'churn_label': churn_label
    })
    
    # Save to disk if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[DataGenerator] Synthetic dataset saved to {output_path} ({df.shape[0]} rows, {df['churn_label'].mean():.1%} churn rate)")
        
    return df


def load_or_generate_dataset(data_dir: str = "data") -> pd.DataFrame:
    """
    Checks for an existing Telco CSV file; if not found, generates synthetic data.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Check for existing csv files in data_dir or root
    candidates = [
        os.path.join(data_dir, "telco_customer_churn_synthetic.csv"),
        os.path.join(data_dir, "Telco-Customer-Churn.csv"),
        os.path.join(data_dir, "customer_churn.csv"),
        "Telco-Customer-Churn.csv",
        "customer_churn.csv"
    ]
    
    for candidate in candidates:
        if os.path.exists(candidate):
            print(f"[DataLoader] Found existing dataset at: {candidate}")
            df = pd.read_csv(candidate)
            # Ensure total_charges is numeric if loaded from standard raw Telco CSV
            if 'total_charges' in df.columns:
                df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
            elif 'TotalCharges' in df.columns:
                df['total_charges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            return df
            
    print("[DataLoader] No local Telco dataset found. Generating realistic synthetic dataset...")
    synthetic_path = os.path.join(data_dir, "telco_customer_churn_synthetic.csv")
    return generate_synthetic_churn_data(n_samples=7000, output_path=synthetic_path)


if __name__ == "__main__":
    df = load_or_generate_dataset()
    print("Dataset preview:")
    print(df.head())
    print("\nSummary Info:")
    print(df.info())
