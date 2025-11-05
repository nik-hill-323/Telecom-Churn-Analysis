"""
Data Generator for Telecom Customer Churn Analysis
Generates synthetic customer data with realistic patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

class TelecomDataGenerator:
    def __init__(self, n_customers=50000):
        """
        Initialize telecom data generator

        Parameters:
        -----------
        n_customers : int
            Number of customers
        """
        self.n_customers = n_customers

    def generate_customer_data(self):
        """Generate comprehensive customer dataset"""
        print("Generating customer data...")

        customers = []

        for customer_id in range(self.n_customers):
            # Demographics
            age = int(np.random.normal(45, 15))
            age = np.clip(age, 18, 80)

            gender = np.random.choice(['Male', 'Female'])

            # Account information
            tenure_months = int(np.random.exponential(24))
            tenure_months = np.clip(tenure_months, 1, 72)

            # Service usage
            contract_type = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'],
                                            p=[0.5, 0.3, 0.2])

            # Services subscribed
            internet_service = np.random.choice(['No', 'DSL', 'Fiber optic'], p=[0.2, 0.4, 0.4])
            online_security = np.random.choice(['Yes', 'No']) if internet_service != 'No' else 'No'
            online_backup = np.random.choice(['Yes', 'No']) if internet_service != 'No' else 'No'
            device_protection = np.random.choice(['Yes', 'No']) if internet_service != 'No' else 'No'
            tech_support = np.random.choice(['Yes', 'No']) if internet_service != 'No' else 'No'
            streaming_tv = np.random.choice(['Yes', 'No']) if internet_service != 'No' else 'No'
            streaming_movies = np.random.choice(['Yes', 'No']) if internet_service != 'No' else 'No'

            phone_service = np.random.choice(['Yes', 'No'], p=[0.9, 0.1])
            multiple_lines = np.random.choice(['Yes', 'No']) if phone_service == 'Yes' else 'No'

            # Payment
            paperless_billing = np.random.choice(['Yes', 'No'], p=[0.6, 0.4])
            payment_method = np.random.choice([
                'Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'
            ], p=[0.35, 0.2, 0.25, 0.2])

            # Calculate monthly charges based on services
            base_charge = 20
            if internet_service == 'DSL':
                base_charge += 30
            elif internet_service == 'Fiber optic':
                base_charge += 50

            if phone_service == 'Yes':
                base_charge += 15
            if multiple_lines == 'Yes':
                base_charge += 10

            # Add-on services
            add_on_charges = 0
            if online_security == 'Yes':
                add_on_charges += 5
            if online_backup == 'Yes':
                add_on_charges += 5
            if device_protection == 'Yes':
                add_on_charges += 5
            if tech_support == 'Yes':
                add_on_charges += 5
            if streaming_tv == 'Yes':
                add_on_charges += 10
            if streaming_movies == 'Yes':
                add_on_charges += 10

            monthly_charges = base_charge + add_on_charges + np.random.uniform(-5, 10)
            monthly_charges = round(monthly_charges, 2)

            total_charges = monthly_charges * tenure_months + np.random.uniform(-100, 100)
            total_charges = round(max(total_charges, monthly_charges), 2)

            # Customer service interactions
            customer_service_calls = int(np.random.poisson(2))

            # Churn factors
            churn_score = 0

            # Contract type (month-to-month more likely to churn)
            if contract_type == 'Month-to-Month':
                churn_score += 0.3
            elif contract_type == 'One Year':
                churn_score += 0.1

            # Tenure (new customers more likely to churn)
            if tenure_months < 6:
                churn_score += 0.3
            elif tenure_months < 12:
                churn_score += 0.2

            # Payment method (electronic check = higher churn)
            if payment_method == 'Electronic check':
                churn_score += 0.2

            # High charges
            if monthly_charges > 70:
                churn_score += 0.15

            # Customer service calls (dissatisfaction)
            if customer_service_calls > 3:
                churn_score += 0.15

            # Lack of add-on services (less engaged)
            if add_on_charges < 10:
                churn_score += 0.1

            # Senior citizens more likely to churn
            if age > 65:
                churn_score += 0.1

            # Fiber optic with high price
            if internet_service == 'Fiber optic' and monthly_charges > 80:
                churn_score += 0.1

            # Determine churn with randomness
            churned = 1 if (churn_score > 0.5 and np.random.random() < churn_score) else 0

            customers.append({
                'customer_id': f'CUST_{customer_id:06d}',
                'age': age,
                'gender': gender,
                'tenure_months': tenure_months,
                'contract_type': contract_type,
                'phone_service': phone_service,
                'multiple_lines': multiple_lines,
                'internet_service': internet_service,
                'online_security': online_security,
                'online_backup': online_backup,
                'device_protection': device_protection,
                'tech_support': tech_support,
                'streaming_tv': streaming_tv,
                'streaming_movies': streaming_movies,
                'paperless_billing': paperless_billing,
                'payment_method': payment_method,
                'monthly_charges': monthly_charges,
                'total_charges': total_charges,
                'customer_service_calls': customer_service_calls,
                'churned': churned
            })

        return pd.DataFrame(customers)

    def generate_and_save(self):
        """Generate all data and save to files"""
        # Generate dataset
        customers_df = self.generate_customer_data()

        # Save data
        output_dir = '../data/raw'
        os.makedirs(output_dir, exist_ok=True)

        customers_df.to_csv(f'{output_dir}/telecom_customers.csv', index=False)

        print(f"\n=== Data Generation Complete ===")
        print(f"Total customers: {len(customers_df):,}")
        print(f"Churned customers: {customers_df['churned'].sum():,}")
        print(f"Churn rate: {customers_df['churned'].mean()*100:.2f}%")
        print(f"\nFiles saved to: {output_dir}/")

        # Summary statistics
        print("\n=== Customer Summary ===")
        print(f"Average age: {customers_df['age'].mean():.1f}")
        print(f"Average tenure: {customers_df['tenure_months'].mean():.1f} months")
        print(f"Average monthly charges: ${customers_df['monthly_charges'].mean():.2f}")
        print(f"Average total charges: ${customers_df['total_charges'].mean():.2f}")

        print("\nContract type distribution:")
        print(customers_df['contract_type'].value_counts())

        print("\nInternet service distribution:")
        print(customers_df['internet_service'].value_counts())

        return customers_df


if __name__ == "__main__":
    generator = TelecomDataGenerator(n_customers=50000)
    customers_df = generator.generate_and_save()
