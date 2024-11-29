import pandas as pd

def load_and_process_data(men_non_t2dm_path, men_t2dm_path, women_non_t2dm_path, women_t2dm_path):
    """
    Load datasets from CSV files for men and women and process them separately for T2DM and non-T2DM groups.
    Returns separate DataFrames for features (X) and labels (y) for both genders.
    """
    # Load the datasets
    men_non_t2dm_data = pd.read_csv(men_non_t2dm_path)
    men_t2dm_data = pd.read_csv(men_t2dm_path)
    women_non_t2dm_data = pd.read_csv(women_non_t2dm_path)
    women_t2dm_data = pd.read_csv(women_t2dm_path)
    
    # Combine data for men and women separately
    men_data = pd.concat([men_non_t2dm_data, men_t2dm_data])
    women_data = pd.concat([women_non_t2dm_data, women_t2dm_data])
    
    # Separate features and labels for men
    X_men = men_data[['Intensity', 'APQ11_Shimmer']]
    y_men = men_data['Diagnosis_Label']
    
    # Separate features and labels for women
    X_women = women_data[['Pitch', 'Pitch_SD', 'RAP_Jitter']]
    y_women = women_data['Diagnosis_Label']
    
    return X_men, y_men, X_women, y_women

# Example usage
if __name__ == "__main__":
    # File paths for each group
    men_non_t2dm_path = 'men_non_t2dm.csv'
    men_t2dm_path = 'men_t2dm.csv'
    women_non_t2dm_path = 'women_non_t2dm.csv'
    women_t2dm_path = 'women_t2dm.csv'
    
    # Load and process the data
    X_men, y_men, X_women, y_women = load_and_process_data(men_non_t2dm_path, men_t2dm_path, women_non_t2dm_path, women_t2dm_path)
    
    # Display the first few rows of the datasets
    print("Men's Data Features:")
    print(X_men.head())
    print("\nMen's Data Labels:")
    print(y_men.head())
    
    print("\nWomen's Data Features:")
    print(X_women.head())
    print("\nWomen's Data Labels:")
    print(y_women.head())
