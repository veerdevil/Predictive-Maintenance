# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/VeerendraManikonda/predictive-maintenance-engine-health-dataset/engine_data.csv.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

#Rename Columns
df.columns = [
    'Engine_RPM',
    'Lub_Oil_Pressure',
    'Fuel_Pressure',
    'Coolant_Pressure',
    'Lub_Oil_Temperature',
    'Coolant_Temperature',
    'Engine_Condition'
]

# Assuming 'data' is our DataFrame and 'features' are our columns of interest
features = df.columns[:-1]  # Exclude the target variable 'Engine Condition'

# Initialize the RobustScaler
scaler = RobustScaler()

# Fit the scaler to the features and transform
data_scaled = scaler.fit_transform(df[features])

# Convert the array back to a DataFrame
data_scaled = pd.DataFrame(data_scaled, columns=features)

# Optionally, we can add the target variable back to the scaled DataFrame
data_scaled['Engine_Condition'] = df['Engine_Condition']

# Now 'data_scaled' is a DataFrame with the scaled features

# Define the target variable
target = 'Engine_Condition'

features = df.select_dtypes(include=np.number).columns.tolist()

# Define X and y
X = data_scaled.drop('Engine_Condition', axis=1)
y = data_scaled['Engine_Condition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train.to_csv("X_train.csv",index=False)
X_test.to_csv("X_test.csv",index=False)
y_train.to_csv("y_train.csv",index=False)
y_test.to_csv("y_test.csv",index=False)


files = ["X_train.csv","X_test.csv","y_train.csv","y_test.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="VeerendraManikonda/predictive-maintenance-engine-health-dataset",
        repo_type="dataset",
    )
