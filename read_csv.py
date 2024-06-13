
import pandas as pd

# Path to your CSV file
file_path = "/home/abood/FINALPROJECT/df_train_cleaned.csv"

# Try to read the CSV file
try:
    df_train_cleaned = pd.read_csv(file_path)
    print(df_train_cleaned.head())  # Display the first few rows of the dataframe
    print(len(df_train_cleaned))
except FileNotFoundError:
    print("The file was not found. Please check the file path and try again.")
except Exception as e:
    print(f"An error occurred: {e}")
