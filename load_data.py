from datasets import load_dataset

# Load the Ecommerce dataset
dataset = load_dataset("bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset")

# Convert the dataset to a pandas DataFrame
import pandas as pd
df = pd.DataFrame(dataset['train'])

# Select only the 'instruction' and 'response' columns
df_selected = df[['instruction', 'response']]

# Save the selected DataFrame as a CSV file
output_path = "ecommerce_dataset.csv"
top_100_df = df_selected.head(100)
top_100_df.to_csv(output_path, index=False)

print(f"Dataset saved to {output_path}")
