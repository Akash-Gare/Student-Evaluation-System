import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load data
df = pd.read_csv("data/student-mat.csv", sep=";")

# 2. Create target
df["pass"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

# 3. Remove leakage
df = df.drop(["G1", "G2", "G3"], axis=1)

# 4. Split features & target
X = df.drop("pass", axis=1)
y = df["pass"]

# 5. Identify column types
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

# 6. Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Apply preprocessing
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

# 9. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_p, y_train)

# 10. Evaluate
y_pred = model.predict(X_test_p)
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", acc)

# 11. Save model & preprocessor
joblib.dump(model, "model/model.pkl")
joblib.dump(preprocessor, "model/preprocessor.pkl")

print("Model and preprocessor saved successfully!")
