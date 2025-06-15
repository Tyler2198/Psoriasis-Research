### Package Import 
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
import statsmodels.api as sm

### Data Import and Pre-Process
df = pd.read_excel("/Data_Path", sheet_name="PsoriasisPanel")
df = df.iloc[:280450]
columns = pd.read_excel("/Users/kryptonempyrean/Desktop/Tesi Material/OneDrive_1_08-09-2024/Psoriasis_2017_Erez_Data2_Coded codifica in corso.xlsx", sheet_name="Foglio1", header=None)
columns = columns[1].tolist()

comorbidities = columns[:142]
selected_comorbidities = sorted(set(df.columns.tolist()) & set(comorbidities),  key = df.columns.tolist().index)
selected_comorbidities.remove("Psoriasis")

selected_columns = ['date_of_birth','GroupName','age','sex','Start_D','DiagYear', "Infertility ", "Psoriasis"] + selected_comorbidities

data = df[selected_columns]
data_copied = data.copy()

### Toddlers Analysis (0-2 years old)
data_toddler = data_copied[(0 <= data_copied['age']) & (data_copied['age'] <= 2)].copy()
data_toddler[data_toddler['sex'] == 'M']

for col in selected_comorbidities:
    data_toddler[col] = data_toddler[col].notna().astype(int)

data_toddler['Psoriasis'] = (data_toddler['GroupName'].str.lower() == 'psoriasis').astype(int)

results = []

for comorb in selected_comorbidities:
    try:
        # Remove rows with NaNs in either Psoriasis or the comorbidity
        subset = data_toddler[['Psoriasis', 'sex', comorb]].dropna()

        if subset[comorb].nunique() < 2:
            continue  # skip comorbidities that are all 0 or all 1
        
        X = sm.add_constant(subset[comorb])
        y = subset['Psoriasis']

        model = sm.Logit(y, X)
        result = model.fit(disp=0)

        or_val = np.exp(result.params[comorb])
        conf_int = result.conf_int().loc[comorb]
        lower, upper = np.exp(conf_int[0]), np.exp(conf_int[1])
        pval = result.pvalues[comorb]
        prevalence = subset[comorb][subset['Psoriasis'] == 1].mean()


        results.append({
            'Comorbidity': comorb,
            'OR': or_val,
            'CI_lower': lower,
            'CI_upper': upper,
            'p_value': pval,
            'Prevalence': prevalence,
            'PsO_Count': subset[comorb][subset['Psoriasis'] == 1].sum(),
            'Female_PsO_Count': subset[comorb][(subset['Psoriasis'] == 1) & (subset['sex'] == 'F')].sum(),
            'Male_PsO_Count': subset[comorb][(subset['Psoriasis'] == 1) & (subset['sex'] == 'M')].sum()
        })
    except Exception as e:
        print(f"Skipped {comorb}: {e}")

  results_df = pd.DataFrame(results)

significant_df = results_df[results_df['p_value'] < 0.05].copy()     # Filter only significant comorbidities (p < 0.05)
significant_df['Distance'] = 1 / significant_df['OR']

# Prepare data
significant_df = results_df[(results_df['p_value'] < 0.05) & (results_df['OR'] > 0)].copy()
nonsignificant_df = results_df[(results_df['p_value'] >= 0.05) & (results_df['OR'] > 0)].copy()

# Compute distance from center
significant_df['Distance'] = 1 / significant_df['OR']
nonsignificant_df['Distance'] = 1 / nonsignificant_df['OR']

significant_df = significant_df[(significant_df['Distance'] <= 1) & (significant_df['Distance'] > 0.0001)]
nonsignificant_df = nonsignificant_df[(nonsignificant_df['Distance'] <= 1) & (nonsignificant_df['Distance'] > 0.0001)]

# Assign angles evenly
n_sig = len(significant_df)
n_nonsig = len(nonsignificant_df)

sig_angles = np.linspace(0, 2 * np.pi, n_sig, endpoint=False)
nonsig_angles = np.linspace(0, 2 * np.pi, n_nonsig, endpoint=False)

significant_df['x'] = significant_df['Distance'] * np.cos(sig_angles)
significant_df['y'] = significant_df['Distance'] * np.sin(sig_angles)

nonsignificant_df['x'] = nonsignificant_df['Distance'] * np.cos(nonsig_angles)
nonsignificant_df['y'] = nonsignificant_df['Distance'] * np.sin(nonsig_angles)

# ---- PLOTTING ----
fig, ax = plt.subplots(figsize=(10, 10))

# Plot non-significant bubbles (gray)
for _, row in nonsignificant_df.iterrows():
    radius = np.sqrt(row['Prevalence'])*100   # Adjust 50 as needed
    size = np.pi * radius ** 2
    ax.scatter(row['x'], row['y'],
               s=size,
               color='lightgray',
               edgecolors='gray',
               linewidths=0.5,
               alpha=0.6)
    ax.text(row['x'], row['y'],
            row['Comorbidity'],
            fontsize=8,
            ha='center',
            va='center',
            rotation=45)


# Plot significant bubbles (colored)
for _, row in significant_df.iterrows():
    radius = np.sqrt(row['Prevalence'])*100  # Adjust 50 as needed
    size = np.pi * radius ** 2
    ax.scatter(row['x'], row['y'],
               s=size,
               alpha=0.8,
               color='#e41a1c' if row['OR'] > 1 else '#4daf4a',
               edgecolors='black',
               linewidths=0.5)
    ax.text(row['x'], row['y'],
            row['Comorbidity'],
            fontsize=8,
            ha='center',
            va='center',
            rotation=45)

# Add dashed circle at OR = 1 (Distance = 1)
circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=1.5)
ax.add_artist(circle)

# Psoriasis marker at center
ax.scatter(0, 0, marker='D', color='black', zorder=3)
ax.text(0, 0, 'Psoriasis', fontsize=11, fontweight='bold',
        color='black', ha='center', va='center', zorder=4)

# Layout settings
max_dist = max(significant_df['Distance'].max(), nonsignificant_df['Distance'].max()) + 0.5
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.show()

### Early Childhood (3-5 years old)
data_EC = data_copied[(3 <= data_copied['age']) & (data_copied['age'] <= 5)].copy()

for col in selected_comorbidities:
    data_EC[col] = data_EC[col].notna().astype(int)

data_EC['Psoriasis'] = (data_EC['GroupName'].str.lower() == 'psoriasis').astype(int)

results = []

for comorb in selected_comorbidities:
    try:
        # Remove rows with NaNs in either Psoriasis or the comorbidity
        subset = data_EC[['Psoriasis', 'sex', comorb]].dropna()

        if subset[comorb].nunique() < 2:
            continue  # skip comorbidities that are all 0 or all 1
        
        X = sm.add_constant(subset[comorb])
        y = subset['Psoriasis']

        model = sm.Logit(y, X)
        result = model.fit(disp=0)

        or_val = np.exp(result.params[comorb])
        conf_int = result.conf_int().loc[comorb]
        lower, upper = np.exp(conf_int[0]), np.exp(conf_int[1])
        pval = result.pvalues[comorb]
        prevalence = subset[comorb][subset['Psoriasis'] == 1].mean()


        results.append({
            'Comorbidity': comorb,
            'OR': or_val,
            'CI_lower': lower,
            'CI_upper': upper,
            'p_value': pval,
            'Prevalence': prevalence,
            'PsO_Count': subset[comorb][subset['Psoriasis'] == 1].sum(),
            'Female_PsO_Count': subset[comorb][(subset['Psoriasis'] == 1) & (subset['sex'] == 'F')].sum(),
            'Male_PsO_Count': subset[comorb][(subset['Psoriasis'] == 1) & (subset['sex'] == 'M')].sum()
        })
    except Exception as e:
        print(f"Skipped {comorb}: {e}")


results_df = pd.DataFrame(results)

# Filter only significant comorbidities (p < 0.05)
significant_df = results_df[results_df['p_value'] < 0.05].copy()
significant_df['Distance'] = 1 / significant_df['OR']
results_df['Distance'] = 1 / results_df['OR']


# Prepare data
significant_df = results_df[(results_df['p_value'] < 0.05) & (results_df['OR'] > 0)].copy()
nonsignificant_df = results_df[(results_df['p_value'] >= 0.05) & (results_df['OR'] > 0)].copy()

# Compute distance from center
significant_df['Distance'] = 1 / significant_df['OR']
nonsignificant_df['Distance'] = 1 / nonsignificant_df['OR']

significant_df = significant_df[(significant_df['Distance'] <= 1) & (significant_df['Distance'] > 0.0001)]
nonsignificant_df = nonsignificant_df[(nonsignificant_df['Distance'] <= 1) & (nonsignificant_df['Distance'] > 0.0001)]

# Assign angles evenly
n_sig = len(significant_df)
n_nonsig = len(nonsignificant_df)

sig_angles = np.linspace(0, 2 * np.pi, n_sig, endpoint=False)
nonsig_angles = np.linspace(0, 2 * np.pi, n_nonsig, endpoint=False)

significant_df['x'] = significant_df['Distance'] * np.cos(sig_angles)
significant_df['y'] = significant_df['Distance'] * np.sin(sig_angles)

nonsignificant_df['x'] = nonsignificant_df['Distance'] * np.cos(nonsig_angles)
nonsignificant_df['y'] = nonsignificant_df['Distance'] * np.sin(nonsig_angles)

# ---- PLOTTING ----
fig, ax = plt.subplots(figsize=(10, 10))

# Plot non-significant bubbles (gray)
for _, row in nonsignificant_df.iterrows():
    radius = np.sqrt(row['Prevalence'])*100   # Adjust 50 as needed
    size = np.pi * radius ** 2
    ax.scatter(row['x'], row['y'],
               s=size,
               color='lightgray',
               edgecolors='gray',
               linewidths=0.5,
               alpha=0.6)
    ax.text(row['x'], row['y'],
            row['Comorbidity'],
            fontsize=8,
            ha='center',
            va='center',
            rotation=45)


# Plot significant bubbles (colored)
for _, row in significant_df.iterrows():
    radius = np.sqrt(row['Prevalence'])*100  # Adjust 50 as needed
    size = np.pi * radius ** 2
    ax.scatter(row['x'], row['y'],
               s=size,
               alpha=0.8,
               color='#e41a1c' if row['OR'] > 1 else '#4daf4a',
               edgecolors='black',
               linewidths=0.5)
    ax.text(row['x'], row['y'],
            row['Comorbidity'],
            fontsize=8,
            ha='center',
            va='center',
            rotation=45)

# Add dashed circle at OR = 1 (Distance = 1)
circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=1.5)
ax.add_artist(circle)

# Psoriasis marker at center
ax.scatter(0, 0, marker='D', color='black', zorder=3)
ax.text(0, 0, 'Psoriasis', fontsize=11, fontweight='bold',
        color='black', ha='center', va='center', zorder=4)

# Layout settings
max_dist = max(significant_df['Distance'].max(), nonsignificant_df['Distance'].max()) + 0.5
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.show()

### Middle Childhood (6-11 years old)
data_MC= data_copied[(6 <= data_copied['age']) & (data_copied['age'] <= 11)].copy()

for col in selected_comorbidities:
    data_MC[col] = data_MC[col].notna().astype(int)

data_MC['Psoriasis'] = (data_MC['GroupName'].str.lower() == 'psoriasis').astype(int)

results = []

for comorb in selected_comorbidities:
    try:
        # Remove rows with NaNs in either Psoriasis or the comorbidity
        subset = data_MC[['Psoriasis', 'sex', comorb]].dropna()

        if subset[comorb].nunique() < 2:
            continue  # skip comorbidities that are all 0 or all 1
        
        X = sm.add_constant(subset[comorb])
        y = subset['Psoriasis']

        model = sm.Logit(y, X)
        result = model.fit(disp=0)

        or_val = np.exp(result.params[comorb])
        conf_int = result.conf_int().loc[comorb]
        lower, upper = np.exp(conf_int[0]), np.exp(conf_int[1])
        pval = result.pvalues[comorb]
        prevalence = subset[comorb][subset['Psoriasis'] == 1].mean()


        results.append({
            'Comorbidity': comorb,
            'OR': or_val,
            'CI_lower': lower,
            'CI_upper': upper,
            'p_value': pval,
            'Prevalence': prevalence,
            'PsO_Count': subset[comorb][subset['Psoriasis'] == 1].sum(),
            'Female_PsO_Count': subset[comorb][(subset['Psoriasis'] == 1) & (subset['sex'] == 'F')].sum(),
            'Male_PsO_Count': subset[comorb][(subset['Psoriasis'] == 1) & (subset['sex'] == 'M')].sum()
        })
    except Exception as e:
        print(f"Skipped {comorb}: {e}")

  results_df = pd.DataFrame(results)

# Filter only significant comorbidities (p < 0.001)
significant_df = results_df[results_df['p_value'] < 0.05].copy()
significant_df['Distance'] = 1 / significant_df['OR']

# Prepare data
significant_df = results_df[(results_df['p_value'] < 0.05) & (results_df['OR'] > 0)].copy()
nonsignificant_df = results_df[(results_df['p_value'] >= 0.05) & (results_df['OR'] > 0)].copy()

# Compute distance from center
significant_df['Distance'] = 1 / significant_df['OR']
nonsignificant_df['Distance'] = 1 / nonsignificant_df['OR']

significant_df = significant_df[(significant_df['Distance'] <= 1) & (significant_df['Distance'] > 0.0001)]
nonsignificant_df = nonsignificant_df[(nonsignificant_df['Distance'] <= 1) & (nonsignificant_df['Distance'] > 0.0001)]

# Assign angles evenly
n_sig = len(significant_df)
n_nonsig = len(nonsignificant_df)

sig_angles = np.linspace(0, 2 * np.pi, n_sig, endpoint=False)
nonsig_angles = np.linspace(0, 2 * np.pi, n_nonsig, endpoint=False)

significant_df['x'] = significant_df['Distance'] * np.cos(sig_angles)
significant_df['y'] = significant_df['Distance'] * np.sin(sig_angles)

nonsignificant_df['x'] = nonsignificant_df['Distance'] * np.cos(nonsig_angles)
nonsignificant_df['y'] = nonsignificant_df['Distance'] * np.sin(nonsig_angles)

# ---- PLOTTING ----
fig, ax = plt.subplots(figsize=(10, 10))

# Plot non-significant bubbles (gray)
for _, row in nonsignificant_df.iterrows():
    radius = np.sqrt(row['Prevalence'])*100   # Adjust 50 as needed
    size = np.pi * radius ** 2
    ax.scatter(row['x'], row['y'],
               s=size,
               color='lightgray',
               edgecolors='gray',
               linewidths=0.5,
               alpha=0.6)
    ax.text(row['x'], row['y'],
            row['Comorbidity'],
            fontsize=8,
            ha='center',
            va='center',
            rotation=45)


# Plot significant bubbles (colored)
for _, row in significant_df.iterrows():
    radius = np.sqrt(row['Prevalence'])*100  # Adjust 50 as needed
    size = np.pi * radius ** 2
    ax.scatter(row['x'], row['y'],
               s=size,
               alpha=0.8,
               color='#e41a1c' if row['OR'] > 1 else '#4daf4a',
               edgecolors='black',
               linewidths=0.5)
    ax.text(row['x'], row['y'],
            row['Comorbidity'],
            fontsize=8,
            ha='center',
            va='center',
            rotation=45)

# Add dashed circle at OR = 1 (Distance = 1)
circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=1.5)
ax.add_artist(circle)

# Psoriasis marker at center
ax.scatter(0, 0, marker='D', color='black', zorder=3)
ax.text(0, 0, 'Psoriasis', fontsize=11, fontweight='bold',
        color='black', ha='center', va='center', zorder=4)

# Layout settings
max_dist = max(significant_df['Distance'].max(), nonsignificant_df['Distance'].max()) + 0.5
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.show()

### Early Adolescence (12-18 years old)
data_EA= data_copied[(12 <= data_copied['age']) & (data_copied['age'] <= 18)].copy()

for col in selected_comorbidities:
    data_EA[col] = data_EA[col].notna().astype(int)

data_EA['Psoriasis'] = (data_EA['GroupName'].str.lower() == 'psoriasis').astype(int)

results = []

for comorb in selected_comorbidities:
    try:
        # Remove rows with NaNs in either Psoriasis or the comorbidity
        subset = data_EA[['Psoriasis', 'sex', comorb]].dropna()

        if subset[comorb].nunique() < 2:
            continue  # skip comorbidities that are all 0 or all 1
        
        X = sm.add_constant(subset[comorb])
        y = subset['Psoriasis']

        model = sm.Logit(y, X)
        result = model.fit(disp=0)

        or_val = np.exp(result.params[comorb])
        conf_int = result.conf_int().loc[comorb]
        lower, upper = np.exp(conf_int[0]), np.exp(conf_int[1])
        pval = result.pvalues[comorb]
        prevalence = subset[comorb][subset['Psoriasis'] == 1].mean()


        results.append({
            'Comorbidity': comorb,
            'OR': or_val,
            'CI_lower': lower,
            'CI_upper': upper,
            'p_value': pval,
            'Prevalence': prevalence,
            'PsO_Count': subset[comorb][subset['Psoriasis'] == 1].sum(),
            'Female_PsO_Count': subset[comorb][(subset['Psoriasis'] == 1) & (subset['sex'] == 'F')].sum(),
            'Male_PsO_Count': subset[comorb][(subset['Psoriasis'] == 1) & (subset['sex'] == 'M')].sum()
        })
    except Exception as e:
        print(f"Skipped {comorb}: {e}")

  results_df = pd.DataFrame(results)

# Filter only significant comorbidities (p < 0.001)
significant_df = results_df[results_df['p_value'] < 0.05].copy()
significant_df['Distance'] = 1 / significant_df['OR']

# Prepare data
significant_df = results_df[(results_df['p_value'] < 0.05) & (results_df['OR'] > 0)].copy()
nonsignificant_df = results_df[(results_df['p_value'] >= 0.05) & (results_df['OR'] > 0)].copy()

# Compute distance from center
significant_df['Distance'] = 1 / significant_df['OR']
nonsignificant_df['Distance'] = 1 / nonsignificant_df['OR']

significant_df = significant_df[(significant_df['Distance'] <= 1) & (significant_df['Distance'] > 0.0001)]
nonsignificant_df = nonsignificant_df[(nonsignificant_df['Distance'] <= 1) & (nonsignificant_df['Distance'] > 0.0001)]

# Assign angles evenly
n_sig = len(significant_df)
n_nonsig = len(nonsignificant_df)

sig_angles = np.linspace(0, 2 * np.pi, n_sig, endpoint=False)
nonsig_angles = np.linspace(0, 2 * np.pi, n_nonsig, endpoint=False)

significant_df['x'] = significant_df['Distance'] * np.cos(sig_angles)
significant_df['y'] = significant_df['Distance'] * np.sin(sig_angles)

nonsignificant_df['x'] = nonsignificant_df['Distance'] * np.cos(nonsig_angles)
nonsignificant_df['y'] = nonsignificant_df['Distance'] * np.sin(nonsig_angles)

# ---- PLOTTING ----
fig, ax = plt.subplots(figsize=(10, 10))

# Plot non-significant bubbles (gray)
for _, row in nonsignificant_df.iterrows():
    radius = np.sqrt(row['Prevalence'])*100   # Adjust 50 as needed
    size = np.pi * radius ** 2
    ax.scatter(row['x'], row['y'],
               s=size,
               color='lightgray',
               edgecolors='gray',
               linewidths=0.5,
               alpha=0.6)
    ax.text(row['x'], row['y'],
            row['Comorbidity'],
            fontsize=8,
            ha='center',
            va='center',
            rotation=45)


# Plot significant bubbles (colored)
for _, row in significant_df.iterrows():
    radius = np.sqrt(row['Prevalence'])*100  # Adjust 50 as needed
    size = np.pi * radius ** 2
    ax.scatter(row['x'], row['y'],
               s=size,
               alpha=0.8,
               color='#e41a1c' if row['OR'] > 1 else '#4daf4a',
               edgecolors='black',
               linewidths=0.5)
    ax.text(row['x'], row['y'],
            row['Comorbidity'],
            fontsize=8,
            ha='center',
            va='center',
            rotation=45)

# Add dashed circle at OR = 1 (Distance = 1)
circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=1.5)
ax.add_artist(circle)

# Psoriasis marker at center
ax.scatter(0, 0, marker='D', color='black', zorder=3)
ax.text(0, 0, 'Psoriasis', fontsize=11, fontweight='bold',
        color='black', ha='center', va='center', zorder=4)

# Layout settings
max_dist = max(significant_df['Distance'].max(), nonsignificant_df['Distance'].max()) + 0.5
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.show()

### Late Adolescence (19-21 years old)
data_LA= data_copied[(19 <= data_copied['age']) & (data_copied['age'] <= 21)].copy()

for col in selected_comorbidities:
    data_LA[col] = data_LA[col].notna().astype(int)

data_LA['Psoriasis'] = (data_LA['GroupName'].str.lower() == 'psoriasis').astype(int)

results = []

for comorb in selected_comorbidities:
    try:
        # Remove rows with NaNs in either Psoriasis or the comorbidity
        subset = data_LA[['Psoriasis', 'sex', comorb]].dropna()

        if subset[comorb].nunique() < 2:
            continue  # skip comorbidities that are all 0 or all 1
        
        X = sm.add_constant(subset[comorb])
        y = subset['Psoriasis']

        model = sm.Logit(y, X)
        result = model.fit(disp=0)

        or_val = np.exp(result.params[comorb])
        conf_int = result.conf_int().loc[comorb]
        lower, upper = np.exp(conf_int[0]), np.exp(conf_int[1])
        pval = result.pvalues[comorb]
        prevalence = subset[comorb][subset['Psoriasis'] == 1].mean()


        results.append({
            'Comorbidity': comorb,
            'OR': or_val,
            'CI_lower': lower,
            'CI_upper': upper,
            'p_value': pval,
            'Prevalence': prevalence,
            'PsO_Count': subset[comorb][subset['Psoriasis'] == 1].sum(),
            'Female_PsO_Count': subset[comorb][(subset['Psoriasis'] == 1) & (subset['sex'] == 'F')].sum(),
            'Male_PsO_Count': subset[comorb][(subset['Psoriasis'] == 1) & (subset['sex'] == 'M')].sum()
        })
    except Exception as e:
        print(f"Skipped {comorb}: {e}")

  results_df = pd.DataFrame(results)

# Filter only significant comorbidities (p < 0.05)
significant_df = results_df[results_df['p_value'] < 0.05].copy()
significant_df['Distance'] = 1 / significant_df['OR']

# Prepare data
significant_df = results_df[(results_df['p_value'] < 0.05) & (results_df['OR'] > 0)].copy()
nonsignificant_df = results_df[(results_df['p_value'] >= 0.05) & (results_df['OR'] > 0)].copy()

# Compute distance from center
significant_df['Distance'] = 1 / significant_df['OR']
nonsignificant_df['Distance'] = 1 / nonsignificant_df['OR']

significant_df = significant_df[(significant_df['Distance'] <= 1) & (significant_df['Distance'] > 0.0001)]
nonsignificant_df = nonsignificant_df[(nonsignificant_df['Distance'] <= 1) & (nonsignificant_df['Distance'] > 0.0001)]

# Assign angles evenly
n_sig = len(significant_df)
n_nonsig = len(nonsignificant_df)

sig_angles = np.linspace(0, 2 * np.pi, n_sig, endpoint=False)
nonsig_angles = np.linspace(0, 2 * np.pi, n_nonsig, endpoint=False)

significant_df['x'] = significant_df['Distance'] * np.cos(sig_angles)
significant_df['y'] = significant_df['Distance'] * np.sin(sig_angles)

nonsignificant_df['x'] = nonsignificant_df['Distance'] * np.cos(nonsig_angles)
nonsignificant_df['y'] = nonsignificant_df['Distance'] * np.sin(nonsig_angles)

# ---- PLOTTING ----
fig, ax = plt.subplots(figsize=(10, 10))

# Plot non-significant bubbles (gray)
for _, row in nonsignificant_df.iterrows():
    radius = np.sqrt(row['Prevalence'])*100   # Adjust 50 as needed
    size = np.pi * radius ** 2
    ax.scatter(row['x'], row['y'],
               s=size,
               color='lightgray',
               edgecolors='gray',
               linewidths=0.5,
               alpha=0.6)
    ax.text(row['x'], row['y'],
            row['Comorbidity'],
            fontsize=8,
            ha='center',
            va='center',
            rotation=45)


# Plot significant bubbles (colored)
for _, row in significant_df.iterrows():
    radius = np.sqrt(row['Prevalence'])*100  # Adjust 50 as needed
    size = np.pi * radius ** 2
    ax.scatter(row['x'], row['y'],
               s=size,
               alpha=0.8,
               color='#e41a1c' if row['OR'] > 1 else '#4daf4a',
               edgecolors='black',
               linewidths=0.5)
    ax.text(row['x'], row['y'],
            row['Comorbidity'],
            fontsize=8,
            ha='center',
            va='center',
            rotation=45)

# Add dashed circle at OR = 1 (Distance = 1)
circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=1.5)
ax.add_artist(circle)

# Psoriasis marker at center
ax.scatter(0, 0, marker='D', color='black', zorder=3)
ax.text(0, 0, 'Psoriasis', fontsize=11, fontweight='bold',
        color='black', ha='center', va='center', zorder=4)

# Layout settings
max_dist = max(significant_df['Distance'].max(), nonsignificant_df['Distance'].max()) + 0.5
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.show()
