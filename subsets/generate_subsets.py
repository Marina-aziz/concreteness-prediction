# generate_subsets.py

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Load the existing training_set.csv file and create subsets based on concreteness and frequency.
    
    The subsets generated are:
      - extremes: concreteness < 2 or > 4.
      - mid_range: concreteness between 2 and 4 (inclusive).
      - high_freq: frequency between the 75th and 95th percentiles.
      - low_freq: frequency at or below the 25th percentile.

    Additionally, combined subsets:
      - extremes_high_freq: extremes ∩ high_freq.
      - extremes_low_freq: extremes ∩ low_freq.
      - mid_range_high_freq: mid_range ∩ high_freq.
      - mid_range_low_freq: mid_range ∩ low_freq.

    Assumes train_set.csv has columns:
      ['word','col2','concreteness','col4','col5','col6','col7','col8','col9','col10','frequency','score_bin']
    """
    # Load the entire training set.
    df = pd.read_csv('../datasets/en/en_training.csv')
    
    # Ensure correct column names.
    df.columns = ['word','col2','concreteness','col4','col5','col6','col7',
                  'col8','col9','col10','frequency']
    
    # Cast to numeric and drop missing.
    df['concreteness'] = pd.to_numeric(df['concreteness'], errors='coerce')
    df['frequency']   = pd.to_numeric(df['frequency'], errors='coerce')
    df = df.dropna(subset=['concreteness','frequency'])
    
    logging.info("Loaded training set with %d samples", len(df))
    
    # Main subsets by concreteness.
    extremes  = df[(df['concreteness'] < 2) | (df['concreteness'] > 4)]
    mid_range = df[(df['concreteness'] >= 2) & (df['concreteness'] <= 4)]
    
    # Frequency thresholds: 25th, 75th and 95th percentiles.
    freq_25 = df['frequency'].quantile(0.25)
    freq_75 = df['frequency'].quantile(0.75)
    freq_95 = df['frequency'].quantile(0.95)
    logging.info("Frequency thresholds: 25th=%.2f, 75th=%.2f, 95th=%.2f", freq_25, freq_75, freq_95)
    
    # Main subsets by frequency.
    low_freq  = df[df['frequency'] <= freq_25]
    high_freq = df[(df['frequency'] >= freq_75) & (df['frequency'] < freq_95)]
    
    # Combined subsets.
    extremes_low_freq      = extremes[extremes['frequency'] <= freq_25]
    extremes_high_freq     = extremes[(extremes['frequency'] >= freq_75) & (extremes['frequency'] < freq_95)]
    mid_range_low_freq     = mid_range[mid_range['frequency'] <= freq_25]
    mid_range_high_freq    = mid_range[(mid_range['frequency'] >= freq_75) & (mid_range['frequency'] < freq_95)]
    
    # Save each subset.
    extremes.to_csv('extremes.csv', index=False)
    mid_range.to_csv('mid_range.csv', index=False)
    low_freq.to_csv('low_freq.csv', index=False)
    high_freq.to_csv('high_freq.csv', index=False)
    extremes_low_freq.to_csv('extremes_low_freq.csv', index=False)
    extremes_high_freq.to_csv('extremes_high_freq.csv', index=False)
    mid_range_low_freq.to_csv('mid_range_low_freq.csv', index=False)
    mid_range_high_freq.to_csv('mid_range_high_freq.csv', index=False)
    
    logging.info("Subset files generated:")
    logging.info("  extremes.csv: %d samples", len(extremes))
    logging.info("  mid_range.csv: %d samples", len(mid_range))
    logging.info("  low_freq.csv: %d samples", len(low_freq))
    logging.info("  high_freq.csv: %d samples", len(high_freq))
    logging.info("  extremes_low_freq.csv: %d samples", len(extremes_low_freq))
    logging.info("  extremes_high_freq.csv: %d samples", len(extremes_high_freq))
    logging.info("  mid_range_low_freq.csv: %d samples", len(mid_range_low_freq))
    logging.info("  mid_range_high_freq.csv: %d samples", len(mid_range_high_freq))

if __name__ == "__main__":
    main()
