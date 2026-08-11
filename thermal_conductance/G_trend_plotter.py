import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


csv_data = '/home/pcuser/Runs/Cooldown_A17/pixelmapping_laserverified.csv'
df = pd.read_csv(csv_data)

# Clean column names for easier access (remove spaces, special chars)
df.columns = df.columns.str.replace(' ', '_').str.replace('[^A-Za-z0-9_]+', '', regex=True)

print("--- Original DataFrame Head ---")
print(df.head())
print("\nColumn Names:", df.columns.tolist()) # Show cleaned column names

# --- 2. Map 'G_type' Column to Numerical Width ---
# Define the mapping based on your requirements
width_mapping = {
    'no perf (solid)': 50*15,
    '15 um legs': 15,
    '80 um legs': 80
}

# Apply the mapping to create the 'width_um' column
# Note: The column name 'G type' was cleaned to 'G_type'
df['width_um'] = df['G_description'].map(width_mapping)

print("\n--- DataFrame with 'width_um' ---")
print(df[['Pixel', 'G_type', 'width_um']].head())

# --- 3. Define Aspect Ratio Function ---
# This function currently just returns the width.
# You might need to adjust this based on what 'aspect ratio' truly means
# in your context (e.g., width / some_other_length).
# For now, we'll use width as a proxy for aspect ratio as requested.
def calculate_aspect_ratio(width):
  """Calculates aspect ratio based on the leg width."""
  # Simple 1:1 mapping for now, as requested initially.
  # Modify this calculation if needed. Example: return width / constant_height
  if width == 0:
      return 0 # Or perhaps 1, or NaN, depending on desired handling for 'no perf'
  else:
      # Assuming aspect ratio is proportional to width for this example
      return width # Or width / some_other_value if applicable

# Apply the function to create the 'aspect_ratio' column
df['aspect_ratio'] = df['width_um'].apply(calculate_aspect_ratio)

print("\n--- DataFrame with 'aspect_ratio' ---")
print(df[['Pixel', 'G_type', 'width_um', 'aspect_ratio']].head())

# --- 4. Prepare Data for Plotting ---
# Select relevant columns and handle potential missing values
# Ensure the 'G_at_30_mK_pWK' column is numeric, coercing errors to NaN
# Note: Column name 'G at 30 mK (pW/K)' was cleaned to 'G_at_30_mK_pWK'
df['G_at_30_mK_pWK'] = pd.to_numeric(df['G_at_30_mK_pWK'], errors='coerce')

# Drop rows where either aspect_ratio or G is missing for plotting
plot_df = df.dropna(subset=['aspect_ratio', 'G_at_30_mK_pWK'])

# --- 5. Plot G vs Aspect Ratio ---
plt.figure(figsize=(10, 6))
plt.scatter(plot_df['aspect_ratio'], plot_df['G_at_30_mK_pWK'])

# Add labels and title
plt.xlabel("Aspect Ratio (Derived from G type leg width [um])")
plt.ylabel("G at 30 mK (pW/K)")
plt.title("Thermal Conductance (G) vs. Aspect Ratio")
plt.grid(True, linestyle='--', alpha=0.6)

# Optional: Set x-axis ticks to match the discrete width values
plt.xticks(sorted(plot_df['aspect_ratio'].unique()))

# Show the plot
plt.show()