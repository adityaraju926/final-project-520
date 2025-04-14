import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

def create_columns(df):
    """
    Create additional columns based on existing data
    """
    # Handle missing values before creating age groups
    df['Age(years)'] = pd.to_numeric(df['Age(years)'], errors='coerce')
    df['Age(years)'] = df['Age(years)'].fillna(df['Age(years)'].median())
    
    df['Age_Group'] = pd.cut(df['Age(years)'], 
                            bins=[-float('inf'), 18, 35, 50, 65, float('inf')],
                            labels=['0-18', '19-35', '36-50', '51-65', '65+'])
    
    # Handle missing values before creating income groups
    df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
    df['Income'] = df['Income'].fillna(df['Income'].median())
    
    df['Income_Group'] = pd.cut(df['Income'],
                               bins=[-float('inf'), 0, 25000, 50000, 75000, 100000, float('inf')],
                               labels=['<0', '<25k', '25k-50k', '50k-75k', '75k-100k', '>100k'])
    
    return df

def load_and_preprocess_data():
    """
    Load and preprocess the dataset
    """
    df = pd.read_csv('data.csv')
    print(f"Initial data shape: {df.shape}")
    
    obesity_data = df[df['Question'].str.contains('obesity', case=False, na=False)].copy()
    print(f"After filtering obesity data: {obesity_data.shape}")
    
    median_obesity = obesity_data['Data_Value'].median()
    obesity_data['target'] = (obesity_data['Data_Value'] > median_obesity).astype(int)
    
    feature_columns = ['YearStart','LocationAbbr','Sample_Size','Age(years)','Education','Sex','Income','Race/Ethnicity']
    
    X = obesity_data[feature_columns].copy()
    y = obesity_data['target']
    
    original_states = pd.DataFrame({
        'State': obesity_data['LocationAbbr']
    })
    
    categorical_columns = ['LocationAbbr', 'Education', 'Sex', 'Race/Ethnicity']
    for column in categorical_columns:
        X[column] = X[column].fillna('Missing')
        X[column] = LabelEncoder().fit_transform(X[column].astype(str))
    
    X['YearStart'] = pd.to_numeric(X['YearStart'], errors='coerce')
    X['Sample_Size'] = pd.to_numeric(X['Sample_Size'], errors='coerce')
    
    X['Age(years)'] = X['Age(years)'].fillna('0')
    X['Age(years)'] = X['Age(years)'].astype(str).apply(lambda x: x.split('-')[0].strip())
    X['Age(years)'] = pd.to_numeric(X['Age(years)'], errors='coerce')
    
    X['Income'] = X['Income'].fillna('0')
    X['Income'] = X['Income'].astype(str).apply(lambda x: x.split('-')[0].replace('$', '').replace(',', '').strip())
    X['Income'] = pd.to_numeric(X['Income'], errors='coerce')
    
    # Fill missing values in numerical columns with median
    numerical_columns = ['YearStart', 'Sample_Size', 'Age(years)', 'Income']
    for col in numerical_columns:
        median_val = X[col].median()
        if pd.isna(median_val):  
            median_val = 0
        X[col] = X[col].fillna(median_val)
    
    # Create features
    X['Age_Group'] = pd.cut(X['Age(years)'], 
                           bins=[-float('inf'), 18, 35, 50, 65, float('inf')],
                           labels=['0-18', '19-35', '36-50', '51-65', '65+'])
    
    X['Income_Group'] = pd.cut(X['Income'],
                              bins=[-float('inf'), 0, 25000, 50000, 75000, 100000, float('inf')],
                              labels=['<0', '<25k', '25k-50k', '50k-75k', '75k-100k', '>100k'])
    
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    
    # Verify there are no null values remaining
    if X.isna().any().any():
        print("Warning: null values found. Dropping affected rows.")
        mask = X.notna().all(axis=1)
        X = X[mask]
        y = y[mask]
        original_states = original_states[mask]
    
    print(f"Final data shape: {X.shape}")
    
    feature_selector = SelectKBest(score_func=f_classif, k=min(10, len(numerical_columns + categorical_columns)))
    X_selected = feature_selector.fit_transform(X[numerical_columns + categorical_columns], y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=7, stratify=y
    )
    
    # Split the original state data accordingly
    _, original_states_test = train_test_split(
        original_states, test_size=0.2, random_state=7, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, original_states_test 