import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load model and scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Load your test dataset
df = pd.read_csv("cleanedForGraph.csv")  # replace with your actual CSV

# Drop the features not used during training
df = df.drop(columns=['BUI', 'DC', 'day', 'month', 'year'],axis=1)

# Separate features and target
X_test = df.drop(columns=['FWI'])  # replace with your actual target column
y_test = df['FWI']
X_test = X_test.drop(columns=['Unnamed: 0'], errors='ignore')
# Scale
X_test_scaled = scaler.transform(X_test)

# Predict
y_pred = ridge_model.predict(X_test_scaled)

# Plot
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('FWI Ridge Regression Prediction vs Actual')
plt.ylabel('FWI Values')
plt.savefig("static/prediction_plot_.png")  # Save it to display on your Flask app
plt.close()

