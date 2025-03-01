from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import pandas as pd
import io
import json
import plotly.express as px
import matplotlib.pyplot as plt
import re
import base64
from io import BytesIO, StringIO
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
)
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
import seaborn as sns
import logging

app = FastAPI()

# Configure Gemini API Key
genai.configure(api_key="AIzaSyAZp_icM6RRxryPP1zu-guOSd_LMWRSpUU")

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Function to convert base64 string to image
def base64_to_image(base64_string):
    byte_data = base64.b64decode(base64_string)
    return BytesIO(byte_data)


# Function to read uploaded file
async def read_uploaded_file(file: UploadFile):
    content = await file.read()
    if file.filename.endswith(".csv"):
        return pd.read_csv(io.StringIO(content.decode("utf-8")))
    elif file.filename.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(content))
    else:
        raise ValueError("Unsupported file format. Upload CSV or Excel.")


# Global variable for dataframe (consider a better solution for production)
# A database or file storage would be better for production use
df_cache = {}


# New endpoint to retrieve columns from uploaded file
@app.post("/upload-columns")
async def upload_columns(file: UploadFile = File(...)):
    try:
        df = await read_uploaded_file(file)

        # Cache the dataframe with a unique identifier (filename in this case)
        file_id = file.filename
        df_cache[file_id] = df

        # Return columns and a preview of the data
        return {
            "columns": df.columns.tolist(),
            "preview": df.head(5).to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze_data(file: UploadFile = File(...), query: str = Form(...)):
    try:
        df = await read_uploaded_file(file)
        data_json = df.to_json(orient="records")

        # Construct prompt for Gemini
        prompt = f"""
        You are a data analyst. Analyze the given dataset and answer the user's query. Also, provide Python code for visualization that supports the summary.
        Format the response **strictly as valid JSON**, without any markdown formatting or explanations.
        data has already been uploaded in df variable
        User Query: {query}
        Dataset: {df}

        Expected JSON Output:
        {{
            "summary": "<Short text summary along with the explanation of the visualization you made.>",
            "code": "<Python code for visualization using matplotlib>"
        }}

        Ensure:
        - The response is a **valid JSON** (no markdown, no extra text).
        - The code contains this line: `plt.savefig('plot.png')`.
        - Do not use `animation_group` or `return`.
        """

        print("Sending Prompt to Gemini...")
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        llm_response = response.text.strip()

        llm_response = re.sub(r"```json\n|\n```", "", llm_response).strip()
        # Parse and validate LLM response
        llm_output = json.loads(llm_response)

        summary = llm_output["summary"]
        code = llm_output["code"]

        print(summary)

        try:
            # Execute the generated visualization code
            exec(code)

            # Convert the image to Base64
            with open("plot.png", "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        except Exception as e:
            print({"error": str(e)})
            image_base64 = None

        return {
            "summary": summary,
            "visualizations": (
                image_base64 if image_base64 else "Error generating visualization."
            ),
        }

    except Exception as e:
        print("Error:", str(e))
        return {"error": str(e)}


# Updated endpoint for predictive analysis
@app.post("/predictive-analysis")
async def predictive_analysis(
    file: UploadFile = File(...),
    target: str = Form(...),
    features: str = Form(...),
    model: str = Form(...),
):
    try:
        # Parse features (sent as JSON string)
        feature_list = json.loads(features)

        # Read the uploaded file
        df = await read_uploaded_file(file)

        # Check if target and features exist in dataframe
        if target not in df.columns:
            raise HTTPException(
                status_code=400, detail=f"Target column '{target}' not found in dataset"
            )

        for feature in feature_list:
            if feature not in df.columns:
                raise HTTPException(
                    status_code=400, detail=f"Feature '{feature}' not found in dataset"
                )

        # Prepare data
        X = df[feature_list].copy()
        y = df[target].copy()

        # Handle missing values
        numeric_cols = X.select_dtypes(include=["number"]).columns
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns

        if len(numeric_cols) > 0 and X[numeric_cols].isnull().any().any():
            numeric_imputer = SimpleImputer(strategy="mean")
            X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])

        if len(categorical_cols) > 0 and X[categorical_cols].isnull().any().any():
            categorical_imputer = SimpleImputer(strategy="most_frequent")
            X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])

        # Encode categorical features
        encoders = {}
        for col in categorical_cols:
            encoders[col] = LabelEncoder()
            X[col] = encoders[col].fit_transform(X[col])

        # Encode target if it's categorical
        target_encoder = None
        if y.dtype == "object" or y.dtype.name == "category":
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Train model
        if model == "Linear Regression":
            model_instance = LinearRegression()
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)

            # Get coefficients
            coefficients = {}
            for feature, coef in zip(feature_list, model_instance.coef_):
                coefficients[feature] = float(coef)

            response = {
                "mse": float(mean_squared_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
                "coefficients": {
                    "intercept": float(model_instance.intercept_),
                    **coefficients,
                },
            }

        elif model == "Logistic Regression":
            model_instance = LogisticRegression(max_iter=1000)
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)

            response = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classification_report": classification_report(
                    y_test, y_pred, output_dict=True
                ),
            }

        elif model == "Random Forest":
            # Check if classification or regression task
            if len(np.unique(y_train)) <= 10:  # Classification
                model_instance = RandomForestClassifier(random_state=42)
                model_instance.fit(X_train, y_train)
                y_pred = model_instance.predict(X_test)

                response = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                    "classification_report": classification_report(
                        y_test, y_pred, output_dict=True
                    ),
                    "feature_importance": dict(
                        zip(feature_list, model_instance.feature_importances_.tolist())
                    ),
                }
            else:  # Regression
                model_instance = RandomForestRegressor(random_state=42)
                model_instance.fit(X_train, y_train)
                y_pred = model_instance.predict(X_test)

                response = {
                    "mse": float(mean_squared_error(y_test, y_pred)),
                    "r2": float(r2_score(y_test, y_pred)),
                    "feature_importance": dict(
                        zip(feature_list, model_instance.feature_importances_.tolist())
                    ),
                }
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported model type: {model}"
            )

        # Create a visualization
        plt.figure(figsize=(10, 6))

        if model == "Linear Regression":
            # Create a scatter plot of actual vs predicted values
            plt.scatter(y_test, y_pred, alpha=0.5)

            # Add a perfect prediction line
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], "r--")

            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(f"Actual vs Predicted: {target}")

        elif model == "Logistic Regression" or (
            model == "Random Forest" and len(np.unique(y_train)) <= 10
        ):
            # Create a confusion matrix heatmap
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")

        else:  # Random Forest Regression
            # Feature importance plot
            importances = model_instance.feature_importances_
            indices = np.argsort(importances)

            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [feature_list[i] for i in indices])
            plt.xlabel("Feature Importance")
            plt.title("Feature Importance in Random Forest")

        # Save the plot and convert to base64
        img_buf = BytesIO()
        plt.savefig(img_buf, format="png")
        plt.close()
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")

        # Add the visualization to the response
        response["visualization"] = img_base64

        return response

    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diagnostic-analysis")
async def diagnostic_analysis(
    file: UploadFile = File(...), target_variable: str = Form(None)
):
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        numeric_data = df.select_dtypes(include=["number"])

        if numeric_data.empty:
            raise HTTPException(
                status_code=400, detail="No numeric columns found for analysis"
            )

        # Correlation Analysis
        correlation_matrix = numeric_data.corr().to_dict()

        # Anomaly Detection using Z-score
        z_scores = np.abs(zscore(numeric_data, nan_policy="omit"))
        anomalies = df[(z_scores > 3).any(axis=1)]
        anomalies_json = anomalies.head(10).to_json(
            orient="records"
        )  # Limit to 10 anomalies for display

        # Root Cause Analysis (Top correlations)
        root_cause_analysis = {}
        if target_variable and target_variable in numeric_data.columns:
            target_corr = (
                numeric_data.corr()[target_variable]
                .sort_values(ascending=False)
                .to_dict()
            )
            root_cause_analysis = {
                k: v for k, v in target_corr.items() if k != target_variable
            }

        # Create a visualization for correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Correlation Matrix Heatmap")

        # Save the plot and convert to base64
        img_buf = BytesIO()
        plt.savefig(img_buf, format="png")
        plt.close()
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")

        return {
            "correlation_matrix": correlation_matrix,
            "anomalies": json.loads(anomalies_json),
            "root_cause_analysis": root_cause_analysis,
            "visualization": img_base64,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/make-prediction")
async def make_prediction(
    file: UploadFile = File(...),
    target: str = Form(...),
    features: str = Form(...),
    model: str = Form(...),
    prediction_values: str = Form(...),
):
    try:
        # Parse inputs
        feature_list = json.loads(features)
        prediction_data = json.loads(prediction_values)

        # Read the uploaded file
        df = await read_uploaded_file(file)

        # Check if target and features exist in dataframe
        if target not in df.columns:
            raise HTTPException(
                status_code=400, detail=f"Target column '{target}' not found in dataset"
            )

        for feature in feature_list:
            if feature not in df.columns:
                raise HTTPException(
                    status_code=400, detail=f"Feature '{feature}' not found in dataset"
                )

        # Prepare data for training
        X = df[feature_list].copy()
        y = df[target].copy()

        # Handle missing values
        numeric_cols = X.select_dtypes(include=["number"]).columns
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns

        numeric_imputer = None
        if len(numeric_cols) > 0 and X[numeric_cols].isnull().any().any():
            numeric_imputer = SimpleImputer(strategy="mean")
            X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])

        categorical_imputer = None
        if len(categorical_cols) > 0 and X[categorical_cols].isnull().any().any():
            categorical_imputer = SimpleImputer(strategy="most_frequent")
            X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])

        # Encode categorical features
        encoders = {}
        for col in categorical_cols:
            encoders[col] = LabelEncoder()
            X[col] = encoders[col].fit_transform(X[col])

        # Prepare prediction data
        pred_input = pd.DataFrame([prediction_data])

        # Apply the same preprocessing to prediction data
        for col in pred_input.columns:
            # Convert input values to appropriate types
            if col in numeric_cols:
                pred_input[col] = pd.to_numeric(pred_input[col])

            # Handle missing values
            if (
                col in numeric_cols
                and pd.isna(pred_input[col]).any()
                and numeric_imputer
            ):
                pred_input[col] = numeric_imputer.transform(pred_input[[col]])

            if (
                col in categorical_cols
                and pd.isna(pred_input[col]).any()
                and categorical_imputer
            ):
                pred_input[col] = categorical_imputer.transform(pred_input[[col]])

            # Encode categorical features
            if col in categorical_cols:
                # Handle new categories not seen during training
                try:
                    pred_input[col] = encoders[col].transform(pred_input[col])
                except ValueError:
                    # If category is unknown, use the most frequent category
                    most_frequent = encoders[col].transform([df[col].mode()[0]])[0]
                    pred_input[col] = most_frequent

        # Encode target if it's categorical
        target_encoder = None
        is_classification = False
        if y.dtype == "object" or y.dtype.name == "category":
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
            is_classification = True
        elif len(np.unique(y)) <= 10:  # Few unique values can be classification
            is_classification = True

        # Train model on full dataset
        if model == "Linear Regression":
            model_instance = LinearRegression()
            model_instance.fit(X, y)
            prediction = model_instance.predict(pred_input)[0]

            return {
                "prediction_type": "regression",
                "predicted_value": float(prediction),
            }

        elif model == "Logistic Regression":
            model_instance = LogisticRegression(max_iter=1000)
            model_instance.fit(X, y)
            prediction = model_instance.predict(pred_input)[0]
            probabilities = model_instance.predict_proba(pred_input)[0]

            # Convert prediction back to original label if encoded
            if target_encoder:
                prediction = target_encoder.inverse_transform([prediction])[0]
                prob_dict = {
                    target_encoder.inverse_transform([i])[0]: float(prob)
                    for i, prob in enumerate(probabilities)
                }
            else:
                prob_dict = {
                    str(i): float(prob) for i, prob in enumerate(probabilities)
                }

            return {
                "prediction_type": "classification",
                "predicted_value": (
                    prediction if isinstance(prediction, str) else float(prediction)
                ),
                "probabilities": prob_dict,
            }

        elif model == "Random Forest":
            # Check if classification or regression task
            if is_classification:  # Classification
                model_instance = RandomForestClassifier(random_state=42)
                model_instance.fit(X, y)
                prediction = model_instance.predict(pred_input)[0]
                probabilities = model_instance.predict_proba(pred_input)[0]

                # Convert prediction back to original label if encoded
                if target_encoder:
                    prediction = target_encoder.inverse_transform([prediction])[0]
                    prob_dict = {
                        target_encoder.inverse_transform([i])[0]: float(prob)
                        for i, prob in enumerate(probabilities)
                    }
                else:
                    prob_dict = {
                        str(i): float(prob) for i, prob in enumerate(probabilities)
                    }

                return {
                    "prediction_type": "classification",
                    "predicted_value": (
                        prediction if isinstance(prediction, str) else float(prediction)
                    ),
                    "probabilities": prob_dict,
                }
            else:  # Regression
                model_instance = RandomForestRegressor(random_state=42)
                model_instance.fit(X, y)
                prediction = model_instance.predict(pred_input)[0]

                return {
                    "prediction_type": "regression",
                    "predicted_value": float(prediction),
                }
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported model type: {model}"
            )

    except Exception as e:
        print("Error during prediction:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# CUSTOM VISUALIZATION


plt.switch_backend("Agg")

# Set up logging
logging.basicConfig(level=logging.DEBUG)

@app.post("/preview-data/")
async def preview_data(file: UploadFile = File(...)):
    """Reads the uploaded CSV file and returns the first few rows as a preview."""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        df.columns = df.columns.str.strip()

        # Convert numerical columns to numeric types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

        preview = df.head().to_dict(orient="records")
        columns = list(df.columns)

        return {"preview": preview, "columns": columns}

    except Exception as e:
        logging.error(f"Error reading file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")



@app.post("/custom-visualization/")
async def custom_visualization(
    file: UploadFile = File(...),
    x_axis: str = Query(None),
    y_axis: str = Query(None),
    plot_type: str = Query(...),
    colors: str = Query(default="#1f77b4"),
    explode: float = Query(default=0.0),  # For pie chart
    bins: int = Query(default=20),  # For histogram
):
    """Generates different types of visualizations based on user input."""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Convert numerical columns to numeric types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

        x_columns = x_axis.split(",") if x_axis else []
        y_columns = y_axis.split(",") if y_axis else []
        color_list = colors.split(",")

        plt.figure(figsize=(8, 6))
        sns.set_theme()

        # Set the title of the graph
        plt.title(f"{plot_type.capitalize()} Plot")

        if plot_type == "pie":
            if not y_columns:
                return {"error": "Y-axis is required for pie chart"}
            data = df[y_columns[0]].value_counts()
            explode_values = [explode] * len(data)  # Apply explode to all slices
            plt.pie(data, labels=data.index, autopct="%1.1f%%", colors=color_list, explode=explode_values, startangle=90)
            plt.axis("equal")  # Ensure the pie chart is circular
        elif plot_type == "histogram":
            if not y_columns:
                return {"error": "Y-axis is required for histogram"}
            sns.histplot(df[y_columns[0]], color=color_list[0] if color_list else "#1f77b4", kde=True, bins=bins)
        elif plot_type == "bar":
            if not x_columns or not y_columns:
                return {"error": "X-axis and Y-axis are required for bar chart"}
            for i, y_col in enumerate(y_columns):
                sns.barplot(x=df[x_columns[0]], y=df[y_col], color=color_list[i] if i < len(color_list) else "blue", label=y_col)
        elif plot_type == "scatter":
            if not x_columns or not y_columns:
                return {"error": "X-axis and Y-axis are required for scatter plot"}
            for i, y_col in enumerate(y_columns):
                sns.scatterplot(x=df[x_columns[0]], y=df[y_col], color=color_list[i] if i < len(color_list) else "blue", label=y_col)
        elif plot_type == "line":
            if not x_columns or not y_columns:
                return {"error": "X-axis and Y-axis are required for line graph"}
            for i, y_col in enumerate(y_columns):
                sns.lineplot(x=df[x_columns[0]], y=df[y_col], color=color_list[i] if i < len(color_list) else "blue", label=y_col)
        elif plot_type == "box":
            if not x_columns or not y_columns:
                return {"error": "X-axis and Y-axis are required for box plot"}
            for i, y_col in enumerate(y_columns):
                sns.boxplot(x=df[x_columns[0]], y=df[y_col], color=color_list[i] if i < len(color_list) else "blue", label=y_col)
        elif plot_type == "violin":
            if not x_columns or not y_columns:
                return {"error": "X-axis and Y-axis are required for violin plot"}
            for i, y_col in enumerate(y_columns):
                sns.violinplot(x=df[x_columns[0]], y=df[y_col], color=color_list[i] if i < len(color_list) else "blue", label=y_col)
        else:
            return {"error": "Invalid plot type"}

        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return {"image": image_base64}

    except Exception as e:
        logging.error(f"Error generating visualization: {str(e)}")
        return {"error": str(e)}