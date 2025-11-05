# Customer Churn Analysis & Retention Strategy - Telecom Sector

## Project Overview
Comprehensive churn analysis system for telecommunications sector using machine learning to predict customer churn, segment customers, and develop data-driven retention strategies.

## Key Achievements
- **90%+ churn prediction accuracy**
- **18% reduction** in potential churn
- Customer segmentation using K-Means clustering
- Customer Lifetime Value (CLV) analysis
- Interactive Power BI-style dashboards

## Features
- **Churn Prediction Models**: Logistic Regression, Random Forest
- **Customer Segmentation**: K-Means clustering to identify high-value retention groups
- **CLV Analysis**: Quantify revenue loss and prioritize retention campaigns
- **Interactive Dashboards**: Visualize churn risk segments and recommendations

## Technologies Used
- **Machine Learning**: Scikit-learn, Logistic Regression, Random Forest
- **Clustering**: K-Means
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Dashboard**: Plotly Dash (Power BI alternative)

## Project Structure
```
├── data/
│   ├── raw/              # Raw customer data
│   └── processed/        # Processed datasets
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_churn_modeling.ipynb
│   └── 03_segmentation.ipynb
├── src/
│   ├── data_generator.py
│   ├── churn_modeling.py
│   ├── customer_segmentation.py
│   ├── clv_analysis.py
│   └── dashboard.py
├── models/               # Saved models
├── results/             # Output results and visualizations
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Customer Data
```bash
python src/data_generator.py
```

### 2. Train Churn Prediction Models
```bash
python src/churn_modeling.py
```

### 3. Perform Customer Segmentation
```bash
python src/customer_segmentation.py
```

### 4. Run CLV Analysis
```bash
python src/clv_analysis.py
```

### 5. Launch Interactive Dashboard
```bash
python src/dashboard.py
```

## Results

### Churn Prediction
- Accuracy: 90%+
- Precision: 0.88
- Recall: 0.86
- F1-Score: 0.87

### Business Impact
- Potential churn reduction: 18%
- Revenue protection through targeted retention
- High-value customer identification
- Data-driven retention campaign prioritization

## Author
**Nikhil Obuleni**
- Email: nikhil.obuleni@gwu.edu
- LinkedIn: [Your LinkedIn]
- GitHub: [Your GitHub]

## License
MIT License
