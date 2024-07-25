import pandas as pd
import sys
from io import StringIO
from sklearn.impute import SimpleImputer

try:
    # Read data from stdin:
    path_file = sys.stdin.read()
    input_data = pd.read_csv(StringIO(path_file))
    
    # Remove specified columns:
    columns_to_be_removed = ['Timestamp', 'Alkalinity_of_treated_water', 'Turbidity_of_treated_water']
    input_data.drop(columns = columns_to_be_removed, inplace = True)
    
    # Impute missing values:
    imputer = SimpleImputer(strategy='median')
    imputed_data = imputer.fit_transform(input_data)
    water_quality_df = pd.DataFrame(imputed_data, columns=input_data.columns)
    
    # Write the processed data to stdout:
    sys.stdout.write(input_data.to_csv(index = False))
    
except Exception as e:
    sys.stderr.write('An error occurred : {}\n'.format(str(e)))