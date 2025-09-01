import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# set plotly to open charts in the default browser
pio.renderers.default = "browser"

# read dataset
data = pd.read_csv("onlinefraud.csv")  
print("Dataset Sample:\n", data.head())

# check dataset for any null values
print("\nNull Values:\n", data.isnull().sum())

# Exploring transaction type
print("\nTransaction Type Counts:\n", data["type"].value_counts())

# Fix variable name for pie chart
type_counts = data["type"].value_counts()
transactions = type_counts.index
quantity = type_counts.values

# Prepare DataFrame for plotting
pie_data = pd.DataFrame({"Transaction": transactions, "Count": quantity})

# Pie chart
figure = px.pie(
    pie_data, values="Count", names="Transaction", hole=0.5, 
    title="Distribution of Transaction Types"
)
figure.show()

# --- CORRELATION (must run BEFORE mapping isFraud to labels) ---
correlation = data.corr(numeric_only=True)
print("\nCorrelation with isFraud:\n", correlation["isFraud"].sort_values(ascending=False))

# Map transaction types to numbers (fix typo " CASH_IN" -> "CASH_IN")
data["type"] = data["type"].map({
    "CASH_OUT": 1,
    "PAYMENT": 2,
    "CASH_IN": 3,
    "TRANSFER": 4,
    "DEBIT": 5
})

# Keep isFraud numeric for training
y = np.array(data["isFraud"])

# Splitting the data
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)

# Accuracy
print("\nModel Accuracy:", model.score(xtest, ytest))

# Prediction
features = np.array([[4, 9000.60, 9000.60, 0.0]])
prediction = model.predict(features)[0]
print("Predicted Class (0=No Fraud, 1=Fraud):", prediction)
print("Prediction Label:", "Fraud" if prediction == 1 else "No Fraud")
