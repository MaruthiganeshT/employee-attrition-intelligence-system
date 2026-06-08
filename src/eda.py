import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(file_path: str = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv", output_dir: str = "eda_reports"):
    """Perform exploratory data analysis and generate business insights."""
    print("Loading data for EDA...")
    df = pd.read_csv(file_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- BASIC STATISTICS ---")
    print(f"Total Employees: {df.shape[0]}")
    print(f"Total Features: {df.shape[1]}")
    
    attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
    print(f"Attrition Rate: {attrition_rate:.2f}%")
    
    # Insights on missing values
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found.")
    else:
        print(f"Missing values:\n{missing[missing > 0]}")
        
    print("\n--- BUSINESS INSIGHTS ---")
    
    # 1. Attrition by Department
    dept_attrition = df.groupby('Department')['Attrition'].apply(lambda x: (x=='Yes').mean() * 100)
    print("\nAttrition Rate by Department (%):")
    print(dept_attrition)
    
    # 2. Income vs Attrition
    avg_income_attrition = df.groupby('Attrition')['MonthlyIncome'].mean()
    print("\nAverage Monthly Income by Attrition:")
    print(avg_income_attrition)
    
    # 3. Job Satisfaction vs Attrition
    js_attrition = df.groupby('JobSatisfaction')['Attrition'].apply(lambda x: (x=='Yes').mean() * 100)
    print("\nAttrition Rate by Job Satisfaction Level (%):")
    print(js_attrition)
    
    # Visualizations
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Attrition Count
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Attrition', data=df, palette='Set2')
    plt.title('Employee Attrition Count')
    plt.savefig(f"{output_dir}/attrition_count.png")
    plt.close()
    
    # Plot 2: Monthly Income vs Attrition
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, palette='Set2')
    plt.title('Monthly Income Distribution by Attrition')
    plt.savefig(f"{output_dir}/income_vs_attrition.png")
    plt.close()
    
    # Plot 3: Correlation Matrix
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title('Feature Correlation Matrix')
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()
    
    print(f"\nEDA completed. Visualizations saved in '{output_dir}/'.")
    print("Key Insight: Employees with lower Monthly Income and lower Job Satisfaction tend to have higher attrition rates.")

if __name__ == "__main__":
    perform_eda()
