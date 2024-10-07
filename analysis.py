import pandas as pd
from collections import Counter

# Load your CSV data into a DataFrame
df = pd.read_csv('ecommerce_dataset.csv')

# Calculate character lengths for 'instruction' and 'response'
df['instruction_length'] = df['instruction'].apply(len)
df['response_length'] = df['response'].apply(len)

# Function to calculate statistics
def calculate_statistics(lengths):
    min_length = lengths.min()
    max_length = lengths.max()
    median_length = lengths.median()
    mode_length = lengths.mode().iloc[0] if not lengths.mode().empty else None
    return min_length, max_length, median_length, mode_length

# Calculate statistics for 'instruction'
instruction_stats = calculate_statistics(df['instruction_length'])
print(f"Instruction - Min: {instruction_stats[0]}, Max: {instruction_stats[1]}, Median: {instruction_stats[2]}, Mode: {instruction_stats[3]}")

# Calculate statistics for 'response'
response_stats = calculate_statistics(df['response_length'])
print(f"Response - Min: {response_stats[0]}, Max: {response_stats[1]}, Median: {response_stats[2]}, Mode: {response_stats[3]}")
