"""
Customer Segmentation using K-Means Clustering
Identify high-value retention groups
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

np.random.seed(42)

class CustomerSegmentation:
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.pca = None

    def load_data(self):
        """Load customer data"""
        print("Loading customer data...")
        df = pd.read_csv('../data/raw/telecom_customers.csv')
        return df

    def prepare_features(self, df):
        """Prepare features for clustering"""
        print("Preparing features for segmentation...")

        # Numerical features for clustering
        feature_cols = [
            'age', 'tenure_months', 'monthly_charges',
            'total_charges', 'customer_service_calls'
        ]

        # Add binary features
        df['has_internet'] = (df['internet_service'] != 'No').astype(int)
        df['has_phone'] = (df['phone_service'] == 'Yes').astype(int)
        df['has_multiple_lines'] = (df['multiple_lines'] == 'Yes').astype(int)
        df['has_online_security'] = (df['online_security'] == 'Yes').astype(int)
        df['has_tech_support'] = (df['tech_support'] == 'Yes').astype(int)
        df['has_streaming'] = ((df['streaming_tv'] == 'Yes') | (df['streaming_movies'] == 'Yes')).astype(int)
        df['long_term_contract'] = (df['contract_type'].isin(['One Year', 'Two Year'])).astype(int)

        feature_cols.extend([
            'has_internet', 'has_phone', 'has_multiple_lines',
            'has_online_security', 'has_tech_support', 'has_streaming',
            'long_term_contract'
        ])

        X = df[feature_cols]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, feature_cols, df

    def find_optimal_clusters(self, X_scaled):
        """Use elbow method to find optimal number of clusters"""
        print("\nFinding optimal number of clusters...")

        inertias = []
        silhouette_scores = []
        K_range = range(2, 11)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)

            from sklearn.metrics import silhouette_score
            score = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(score)

        # Plot elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(K_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True, alpha=0.3)

        ax2.plot(K_range, silhouette_scores, 'go-')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs('../results', exist_ok=True)
        plt.savefig('../results/optimal_clusters.png', dpi=300, bbox_inches='tight')
        print("Optimal clusters plot saved to: ../results/optimal_clusters.png")

    def perform_clustering(self, X_scaled):
        """Perform K-Means clustering"""
        print(f"\nPerforming K-Means clustering (k={self.n_clusters})...")

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(X_scaled)

        # Calculate silhouette score
        from sklearn.metrics import silhouette_score
        sil_score = silhouette_score(X_scaled, clusters)
        print(f"Silhouette Score: {sil_score:.4f}")

        return clusters

    def analyze_segments(self, df, clusters):
        """Analyze customer segments"""
        print("\n=== Analyzing Customer Segments ===")

        df['cluster'] = clusters

        # Segment profiles
        segment_profiles = df.groupby('cluster').agg({
            'age': 'mean',
            'tenure_months': 'mean',
            'monthly_charges': 'mean',
            'total_charges': 'mean',
            'customer_service_calls': 'mean',
            'churned': ['sum', 'mean', 'count']
        }).round(2)

        segment_profiles.columns = ['_'.join(col).strip() for col in segment_profiles.columns.values]

        print("\nSegment Profiles:")
        print(segment_profiles)

        # Calculate Customer Lifetime Value (CLV) proxy
        df['clv_proxy'] = df['monthly_charges'] * df['tenure_months']

        # Segment characteristics
        for cluster_id in range(self.n_clusters):
            cluster_data = df[df['cluster'] == cluster_id]

            print(f"\n--- Cluster {cluster_id} ---")
            print(f"Size: {len(cluster_data):,} customers ({len(cluster_data)/len(df)*100:.1f}%)")
            print(f"Avg Age: {cluster_data['age'].mean():.1f}")
            print(f"Avg Tenure: {cluster_data['tenure_months'].mean():.1f} months")
            print(f"Avg Monthly Charges: ${cluster_data['monthly_charges'].mean():.2f}")
            print(f"Avg CLV Proxy: ${cluster_data['clv_proxy'].mean():.2f}")
            print(f"Churn Rate: {cluster_data['churned'].mean()*100:.2f}%")
            print(f"Total Churned: {cluster_data['churned'].sum()}")

            # Contract type distribution
            print(f"Contract Types:")
            print(cluster_data['contract_type'].value_counts())

        # Save segment data
        os.makedirs('../data/processed', exist_ok=True)
        df.to_csv('../data/processed/segmented_customers.csv', index=False)

        return df

    def visualize_segments(self, X_scaled, df):
        """Visualize customer segments"""
        print("\nCreating visualizations...")

        # PCA for 2D visualization
        self.pca = PCA(n_components=2)
        X_pca = self.pca.fit_transform(X_scaled)

        fig = plt.figure(figsize=(16, 12))

        # 1. PCA Scatter Plot
        ax1 = plt.subplot(3, 3, 1)
        scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'],
                             cmap='viridis', alpha=0.6, s=20)
        ax1.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax1.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax1.set_title('Customer Segments (PCA)')
        plt.colorbar(scatter, ax=ax1, label='Cluster')

        # 2. Cluster sizes
        ax2 = plt.subplot(3, 3, 2)
        cluster_sizes = df['cluster'].value_counts().sort_index()
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_clusters))
        ax2.bar(cluster_sizes.index, cluster_sizes.values, color=colors, alpha=0.7)
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Number of Customers')
        ax2.set_title('Cluster Sizes')
        for i, v in enumerate(cluster_sizes.values):
            ax2.text(i, v + 100, str(v), ha='center', fontweight='bold')

        # 3. Churn rate by cluster
        ax3 = plt.subplot(3, 3, 3)
        churn_by_cluster = df.groupby('cluster')['churned'].mean() * 100
        ax3.bar(churn_by_cluster.index, churn_by_cluster.values, color=colors, alpha=0.7)
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Churn Rate (%)')
        ax3.set_title('Churn Rate by Cluster')
        ax3.axhline(df['churned'].mean()*100, color='red', linestyle='--', label='Overall Average')
        ax3.legend()

        # 4. Monthly charges by cluster
        ax4 = plt.subplot(3, 3, 4)
        df.boxplot(column='monthly_charges', by='cluster', ax=ax4)
        ax4.set_xlabel('Cluster')
        ax4.set_ylabel('Monthly Charges ($)')
        ax4.set_title('Monthly Charges Distribution by Cluster')
        plt.suptitle('')

        # 5. Tenure by cluster
        ax5 = plt.subplot(3, 3, 5)
        df.boxplot(column='tenure_months', by='cluster', ax=ax5)
        ax5.set_xlabel('Cluster')
        ax5.set_ylabel('Tenure (Months)')
        ax5.set_title('Tenure Distribution by Cluster')
        plt.suptitle('')

        # 6. CLV by cluster
        ax6 = plt.subplot(3, 3, 6)
        clv_by_cluster = df.groupby('cluster')['clv_proxy'].mean()
        ax6.bar(clv_by_cluster.index, clv_by_cluster.values, color=colors, alpha=0.7)
        ax6.set_xlabel('Cluster')
        ax6.set_ylabel('Avg CLV Proxy ($)')
        ax6.set_title('Customer Lifetime Value by Cluster')

        # 7. Contract type distribution
        ax7 = plt.subplot(3, 3, 7)
        contract_pivot = pd.crosstab(df['cluster'], df['contract_type'], normalize='index') * 100
        contract_pivot.plot(kind='bar', stacked=True, ax=ax7, alpha=0.8)
        ax7.set_xlabel('Cluster')
        ax7.set_ylabel('Percentage (%)')
        ax7.set_title('Contract Type Distribution by Cluster')
        ax7.legend(title='Contract Type', bbox_to_anchor=(1.05, 1))
        ax7.set_xticklabels(ax7.get_xticklabels(), rotation=0)

        # 8. High-value segments
        ax8 = plt.subplot(3, 3, 8)
        high_value = df.groupby('cluster').apply(
            lambda x: ((x['monthly_charges'] > df['monthly_charges'].median()) &
                      (x['churned'] == 0)).sum()
        )
        ax8.bar(high_value.index, high_value.values, color=colors, alpha=0.7)
        ax8.set_xlabel('Cluster')
        ax8.set_ylabel('Count')
        ax8.set_title('High-Value Non-Churned Customers')

        # 9. Risk matrix
        ax9 = plt.subplot(3, 3, 9)
        risk_matrix = df.groupby('cluster').agg({
            'monthly_charges': 'mean',
            'churned': 'mean'
        })
        scatter = ax9.scatter(risk_matrix['monthly_charges'],
                             risk_matrix['churned'] * 100,
                             s=cluster_sizes.values / 10,
                             c=range(self.n_clusters),
                             cmap='viridis',
                             alpha=0.7)
        for idx, row in risk_matrix.iterrows():
            ax9.annotate(f'C{idx}',
                        (row['monthly_charges'], row['churned']*100),
                        fontsize=12, fontweight='bold')
        ax9.set_xlabel('Avg Monthly Charges ($)')
        ax9.set_ylabel('Churn Rate (%)')
        ax9.set_title('Revenue vs Risk Matrix')
        ax9.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('../results/customer_segmentation.png', dpi=300, bbox_inches='tight')
        print("Segmentation visualizations saved to: ../results/customer_segmentation.png")

    def identify_retention_targets(self, df):
        """Identify high-value retention targets"""
        print("\n=== Identifying Retention Targets ===")

        # High-value customers: high monthly charges, not yet churned
        high_value = df[(df['monthly_charges'] > df['monthly_charges'].quantile(0.75)) &
                       (df['churned'] == 0)]

        # At-risk: high churn probability indicators
        at_risk = high_value[
            (high_value['contract_type'] == 'Month-to-Month') |
            (high_value['customer_service_calls'] > 3) |
            (high_value['tenure_months'] < 12)
        ]

        print(f"\nTotal high-value customers: {len(high_value):,}")
        print(f"High-value at-risk customers: {len(at_risk):,} ({len(at_risk)/len(high_value)*100:.1f}%)")
        print(f"Potential revenue at risk: ${at_risk['monthly_charges'].sum()*12:,.2f}/year")

        # Save retention targets
        at_risk.to_csv('../data/processed/retention_targets.csv', index=False)
        print("\nRetention targets saved to: ../data/processed/retention_targets.csv")

        return at_risk

    def save_model(self):
        """Save clustering model"""
        os.makedirs('../models', exist_ok=True)

        joblib.dump(self.kmeans, '../models/kmeans_model.pkl')
        joblib.dump(self.scaler, '../models/segmentation_scaler.pkl')
        joblib.dump(self.pca, '../models/pca_model.pkl')

        print("\nSegmentation model saved to: ../models/")


def main():
    segmenter = CustomerSegmentation(n_clusters=4)

    # Load data
    df = segmenter.load_data()

    # Prepare features
    X_scaled, feature_cols, df = segmenter.prepare_features(df)

    # Find optimal clusters
    segmenter.find_optimal_clusters(X_scaled)

    # Perform clustering
    clusters = segmenter.perform_clustering(X_scaled)

    # Analyze segments
    df = segmenter.analyze_segments(df, clusters)

    # Visualize
    segmenter.visualize_segments(X_scaled, df)

    # Identify retention targets
    retention_targets = segmenter.identify_retention_targets(df)

    # Save model
    segmenter.save_model()

    print("\n=== Key Achievements ===")
    print("✓ Customer segmentation using K-Means clustering")
    print("✓ High-value retention groups identified")
    print("✓ At-risk customer segments revealed")
    print("✓ Customer Lifetime Value (CLV) proxy calculated")
    print("✓ Retention campaign targets prioritized")


if __name__ == "__main__":
    main()
