import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

class CreditRiskModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.encoders = {}
        self.best_model = None
        
    def create_sample_data(self):
        """Create realistic Indian credit dataset"""
        np.random.seed(42)
        n_samples = 5000
        
        data = {
            'age': np.random.randint(18, 70, n_samples),
            'income': np.random.randint(20000, 500000, n_samples),  # Monthly income in INR
            'loan_amount': np.random.randint(50000, 5000000, n_samples),
            'credit_score': np.random.randint(300, 850, n_samples),
            'employment_years': np.random.uniform(0, 35, n_samples),
            'late_payments': np.random.randint(0, 10, n_samples),
            'existing_debt': np.random.randint(0, 1000000, n_samples),
            'num_credit_cards': np.random.randint(0, 8, n_samples),
            'dependents': np.random.randint(0, 5, n_samples),
            'education': np.random.choice(['Graduate', 'Post Graduate', 'Under Graduate', 'PhD'], n_samples),
            'employment_type': np.random.choice(['Salaried', 'Self-Employed', 'Business', 'Retired'], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Calculate derived features
        df['debt_to_income'] = df['existing_debt'] / df['income']
        df['loan_to_income'] = df['loan_amount'] / df['income']
        
        # Create target variable (default risk)
        risk_score = (
            (df['credit_score'] < 600) * 0.3 +
            (df['late_payments'] > 3) * 0.25 +
            (df['debt_to_income'] > 0.5) * 0.2 +
            (df['loan_to_income'] > 3) * 0.15
        )
        
        df['default_risk'] = (risk_score > 0.4).astype(int)
        
        return df
    
    def preprocess(self, df):
        """Preprocess data for training"""
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['education', 'employment_type']
        for col in categorical_cols:
            self.encoders[col] = LabelEncoder()
            df_processed[col] = self.encoders[col].fit_transform(df_processed[col])
        
        # Select features
        feature_cols = ['age', 'income', 'loan_amount', 'credit_score', 'employment_years',
                       'late_payments', 'existing_debt', 'num_credit_cards', 'dependents',
                       'debt_to_income', 'loan_to_income', 'education', 'employment_type']
        
        X = df_processed[feature_cols]
        y = df_processed['default_risk']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, feature_cols
    
    def train(self):
        """Train multiple models and select the best"""
        print("📊 Creating training data...")
        df = self.create_sample_data()
        
        print("🔄 Preprocessing data...")
        X, y, feature_cols = self.preprocess(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Models to train
        models = {
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        results = {}
        
        print("🤖 Training models...")
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_proba)
            }
            
            self.models[name] = model
            print(f"  {name}: Accuracy={results[name]['accuracy']:.2%}, AUC={results[name]['auc']:.3f}")
        
        # Select best model
        best_name = max(results, key=lambda x: results[x]['accuracy'])
        self.best_model = self.models[best_name]
        
        print(f"\n✅ Best model: {best_name} with {results[best_name]['accuracy']:.2%} accuracy")
        
        # Save model and preprocessors
        os.makedirs('./data/trained_models', exist_ok=True)
        joblib.dump(self.best_model, './data/trained_models/credit_model.pkl')
        joblib.dump(self.scaler, './data/trained_models/scaler.pkl')
        joblib.dump(self.encoders, './data/trained_models/encoders.pkl')
        joblib.dump(feature_cols, './data/trained_models/feature_cols.pkl')
        
        return results, best_name
    
    def predict(self, features_dict):
        """Make prediction for single application"""
        import pandas as pd
        
        # Convert dict to DataFrame
        df = pd.DataFrame([features_dict])
        
        # Calculate derived features
        df['debt_to_income'] = df['existing_debt'] / (df['income'] + 1)
        df['loan_to_income'] = df['loan_amount'] / (df['income'] + 1)
        
        # Encode categorical
        for col, encoder in self.encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])
        
        # Select features
        feature_cols = joblib.load('./data/trained_models/feature_cols.pkl')
        X = df[feature_cols]
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        default_prob = self.best_model.predict_proba(X_scaled)[0, 1]
        creditworthiness = (1 - default_prob) * 100
        
        return {
            'creditworthiness': round(creditworthiness, 2),
            'default_probability': round(default_prob * 100, 2),
            'risk_level': 'High' if default_prob > 0.5 else 'Medium' if default_prob > 0.3 else 'Low'
        }

# Initialize and train model
def initialize_model():
    model = CreditRiskModel()
    try:
        # Try to load existing model
        model.best_model = joblib.load('./data/trained_models/credit_model.pkl')
        model.scaler = joblib.load('./data/trained_models/scaler.pkl')
        model.encoders = joblib.load('./data/trained_models/encoders.pkl')
        print("✅ Loaded existing model")
    except:
        print("📚 Training new model...")
        model.train()
    
    return model

# Global model instance
credit_model = None

def get_model():
    global credit_model
    if credit_model is None:
        credit_model = initialize_model()
    return credit_model