# preprosses-train

# Load and preprocess the data
data = pd.read_csv('preprocessed_train_data_normalized.csv')
data['label'] = data['label'].astype(int)  # convert label column to integer
X = data.drop('label', axis=1)  # features
X.columns = range(X.shape[1])  # set feature names to integers
y = data['label']  # target variable

# Train the random forest classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X, y)
