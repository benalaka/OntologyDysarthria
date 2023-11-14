import os
import pandas as pd

# Define the directories and file names
directory_emotion = 'Log_PhD/data_emotion'
directory_triples = 'Log_PhD/ontology'
directory_dialogue_acts = 'Log_PhD/data_subject'

file_A = 'emotion.csv'
file_B = 'all_triples_without_EDA.csv'
file_C = 'subject.csv'

# Read CSV files into pandas DataFrames
df_A = pd.read_csv(os.path.join(directory_emotion, file_A))
df_B = pd.read_csv(os.path.join(directory_triples, file_B))
df_C = pd.read_csv(os.path.join(directory_dialogue_acts, file_C))

# Merge DataFrames based on the common column "id"
merged_df = pd.merge(df_A, df_B, on='id', how='inner')
merged_df = pd.merge(merged_df, df_C, on='id', how='inner')

# Save the merged DataFrame to a new CSV file
output_directory = 'Log_PhD/dynamic_workspace'
output_file = 'training_big_dataset.csv'
output_path = os.path.join(output_directory, output_file)

merged_df.to_csv(output_path, index=False)

print(f'Merged data saved to: {output_path}')
