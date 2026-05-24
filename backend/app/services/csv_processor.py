import pandas as pd
import numpy as np
from typing import Dict, List, Any

def process_csv_file(df: pd.DataFrame) -> Dict[str, Any]:
    """Process CSV data for batch predictions"""
    
    # Required columns
    required_cols = ['age', 'monthly_income', 'loan_amount', 'credit_score']
    
    # Validate columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Clean data
    df = df.dropna(subset=required_cols)
    
    # Calculate risk scores (simplified for demo)
    risk_scores = []
    for _, row in df.iterrows():
        score = 100
        if row.get('credit_score', 650) < 600:
            score -= 25
        if row.get('late_payments', 0) > 3:
            score -= 20
        if row.get('existing_debt', 0) / (row.get('monthly_income', 1) + 1) > 0.5:
            score -= 15
        risk_scores.append(max(0, min(100, score)))
    
    df['risk_score'] = risk_scores
    df['risk_level'] = pd.cut(df['risk_score'], bins=[0, 40, 70, 101], labels=['High', 'Medium', 'Low'])
    
    return {
        'total_records': len(df),
        'high_risk': int((df['risk_level'] == 'High').sum()),
        'medium_risk': int((df['risk_level'] == 'Medium').sum()),
        'low_risk': int((df['risk_level'] == 'Low').sum()),
        'avg_risk_score': round(df['risk_score'].mean(), 2)
    }