import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load the Dataset ---
df = pd.read_csv('visa_applications.csv')

# --- 1. Get a High-Level Overview ---
print("--- Data Info ---")
df.info()

print("\n--- Numerical Summary ---")
print(df.describe())


# --- 2. Analyze the Target Variable ---
plt.figure(figsize=(8, 6))
sns.countplot(x='visa_approved', data=df)
plt.title('Distribution of Visa Approvals (0 = Rejected, 1 = Approved)')
plt.xlabel('Visa Approval Status')
plt.ylabel('Number of Applicants')
plt.show()


# --- 3. Analyze a Key Feature: Education Level vs. Approval ---
plt.figure(figsize=(10, 7))
sns.countplot(x='education_level', hue='visa_approved', data=df, order=['High School', 'Bachelors', 'Masters', 'PhD'])
plt.title('Visa Approval Status by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Number of Applicants')
plt.legend(title='Status', labels=['Rejected', 'Approved'])
plt.show()

# --- 4. Analyze a Numerical Feature: Monthly Income ---
plt.figure(figsize=(10, 7))
sns.histplot(data=df, x='monthly_income_usd', hue='visa_approved', kde=True, bins=30)
plt.title('Distribution of Monthly Income by Approval Status')
plt.xlabel('Monthly Income (USD)')
plt.ylabel('Number of Applicants')
plt.legend(title='Status', labels=['Approved', 'Rejected'])
plt.show()