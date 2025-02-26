import React, { useState, useEffect } from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";

function PredictiveAnalysis() {
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [preview, setPreview] = useState([]);
  const [targetVariable, setTargetVariable] = useState("");
  const [features, setFeatures] = useState([]);
  const [selectedModel, setSelectedModel] = useState("Linear Regression");
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [prediction, setPrediction] = useState({});
  const [predictionResult, setPredictionResult] = useState(null);

  const { getRootProps, getInputProps } = useDropzone({
    accept: { "text/csv": [".csv"] },
    onDrop: async (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setFile(acceptedFiles[0]);
        // Automatically upload file when dropped
        await handleUpload(acceptedFiles[0]);
      }
    },
  });

  const handleUpload = async (selectedFile) => {
    if (!selectedFile) {
      setError("Please select a file first.");
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      // Send file to the backend for processing
      const response = await axios.post("http://localhost:8080/upload-columns", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      console.log("File upload response:", response.data);

      if (response.data.columns) {
        setColumns(response.data.columns);

        // Set preview data if available
        if (response.data.preview) {
          setPreview(response.data.preview);
        }
      } else {
        setError("No columns were returned from the server.");
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      setError("Failed to upload and process the file. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!targetVariable || features.length === 0) {
      setError("Please select a target variable and at least one feature.");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("target", targetVariable);
      formData.append("features", JSON.stringify(features));
      formData.append("model", selectedModel);

      const response = await axios.post("http://localhost:8080/predictive-analysis", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      console.log("Analysis results:", response.data);
      setResults(response.data);

      // Initialize prediction state with feature fields
      const initialPrediction = {};
      features.forEach(feature => {
        initialPrediction[feature] = "";
      });
      setPrediction(initialPrediction);
      setPredictionResult(null);
    } catch (error) {
      console.error("Error analyzing data:", error);
      setError("Failed to analyze data. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handlePredictionInput = (feature, value) => {
    setPrediction(prev => ({
      ...prev,
      [feature]: value
    }));
  };

  const handlePredict = async () => {
    if (Object.values(prediction).some(val => val === "")) {
      setError("Please fill in all feature values for prediction.");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("target", targetVariable);
      formData.append("features", JSON.stringify(features));
      formData.append("model", selectedModel);
      formData.append("prediction_values", JSON.stringify(prediction));

      const response = await axios.post("http://localhost:8080/make-prediction", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      console.log("Prediction response:", response.data);
      setPredictionResult(response.data);
    } catch (error) {
      console.error("Error making prediction:", error);
      setError("Failed to make prediction. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* File Upload Area */}
      <div
        {...getRootProps()}
        className="border-2 border-dashed rounded-lg p-6 text-center hover:border-indigo-500 cursor-pointer"
      >
        <input {...getInputProps()} />
        <p className="text-gray-600">{file ? `Selected: ${file.name}` : "Upload a CSV file or drag & drop"}</p>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-md">
          <p className="text-red-600">{error}</p>
        </div>
      )}

      {isLoading && <p className="text-center text-gray-600">Loading...</p>}

      {columns.length > 0 && (
        <div className="space-y-6">
          {/* Column Preview */}
          <div>
            <h2 className="text-lg font-medium text-gray-900 mb-2">Dataset Columns</h2>
            <div className="bg-gray-50 p-3 rounded-lg overflow-x-auto">
              <div className="grid grid-cols-4 gap-2">
                {columns.map((col) => (
                  <div key={col} className="bg-white p-2 rounded border">{col}</div>
                ))}
              </div>
            </div>
          </div>

          <div>
            <h2 className="text-lg font-medium text-gray-900 mb-2">Select Target Variable</h2>
            <select
              className="w-full border rounded-lg p-2"
              value={targetVariable}
              onChange={(e) => setTargetVariable(e.target.value)}
            >
              <option value="">Select</option>
              {columns.map((col) => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
          </div>

          <div>
            <h2 className="text-lg font-medium text-gray-900 mb-2">Select Features</h2>
            <select
              multiple
              className="w-full border rounded-lg p-2 h-32"
              onChange={(e) => setFeatures([...e.target.selectedOptions].map(o => o.value))}
            >
              {columns.filter(col => col !== targetVariable).map((col) => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
            {features.length > 0 && (
              <p className="mt-2 text-sm text-gray-600">
                Selected features: {features.join(", ")}
              </p>
            )}
          </div>

          <div>
            <h2 className="text-lg font-medium text-gray-900 mb-2">Select Model</h2>
            <select
              className="w-full border rounded-lg p-2"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              <option value="Linear Regression">Linear Regression</option>
              <option value="Logistic Regression">Logistic Regression</option>
              <option value="Random Forest">Random Forest</option>
            </select>
          </div>

          <button
            onClick={handleAnalyze}
            className="w-full bg-indigo-600 text-white p-3 rounded-md hover:bg-indigo-700 transition"
            disabled={isLoading}
          >
            {isLoading ? "Analyzing..." : "Analyze Data"}
          </button>

          {results && (
            <div className="space-y-6">
              <div className="bg-gray-50 p-4 rounded-lg">
                <h2 className="text-lg font-medium text-gray-900 mb-2">Model Performance</h2>
                <div className="bg-white p-4 rounded border">
                  {results.mse !== undefined && (
                    <div className="mb-2">
                      <span className="font-medium">Mean Squared Error:</span> {results.mse.toFixed(4)}
                    </div>
                  )}
                  {results.r2 !== undefined && (
                    <div className="mb-2">
                      <span className="font-medium">R² Score:</span> {results.r2.toFixed(4)}
                    </div>
                  )}
                  {results.accuracy !== undefined && (
                    <div className="mb-2">
                      <span className="font-medium">Accuracy:</span> {(results.accuracy * 100).toFixed(2)}%
                    </div>
                  )}
                  {results.coefficients && (
                    <div className="mt-4">
                      <h3 className="text-md font-medium">Coefficients:</h3>
                      <div className="grid grid-cols-2 gap-2 mt-2">
                        <div className="p-2 bg-gray-50 rounded">
                          <span className="font-medium">Intercept:</span> {results.coefficients.intercept.toFixed(4)}
                        </div>
                        {Object.entries(results.coefficients)
                          .filter(([key]) => key !== "intercept")
                          .map(([feature, coef]) => (
                            <div key={feature} className="p-2 bg-gray-50 rounded">
                              <span className="font-medium">{feature}:</span> {coef.toFixed(4)}
                            </div>
                          ))}
                      </div>
                    </div>
                  )}
                  {results.feature_importance && (
                    <div className="mt-4">
                      <h3 className="text-md font-medium">Feature Importance:</h3>
                      <div className="grid grid-cols-2 gap-2 mt-2">
                        {Object.entries(results.feature_importance)
                          .sort((a, b) => b[1] - a[1])
                          .map(([feature, importance]) => (
                            <div key={feature} className="p-2 bg-gray-50 rounded">
                              <span className="font-medium">{feature}:</span> {importance.toFixed(4)}
                            </div>
                          ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {results.visualization && (
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h2 className="text-lg font-medium text-gray-900 mb-2">Visualization</h2>
                  <div className="flex justify-center">
                    <img
                      src={`data:image/png;base64,${results.visualization}`}
                      alt="Model Visualization"
                      className="max-w-full h-auto"
                    />
                  </div>
                </div>
              )}

              <div className="bg-gray-50 p-4 rounded-lg">
                <h2 className="text-lg font-medium text-gray-900 mb-2">Make Prediction</h2>
                <div className="space-y-4">
                  {features.map(feature => (
                    <div key={feature} className="flex flex-col">
                      <label className="mb-1 text-sm font-medium text-gray-700">{feature}</label>
                      <input
                        type="text"
                        className="border rounded-lg p-2"
                        value={prediction[feature] || ''}
                        onChange={(e) => handlePredictionInput(feature, e.target.value)}
                        placeholder={`Enter value for ${feature}`}
                      />
                    </div>
                  ))}
                  <button
                    onClick={handlePredict}
                    className="w-full bg-indigo-600 text-white p-3 rounded-md hover:bg-indigo-700 transition mt-4"
                    disabled={isLoading}
                  >
                    {isLoading ? "Predicting..." : "Make Prediction"}
                  </button>
                </div>
              </div>

              {predictionResult && (
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h2 className="text-lg font-medium text-gray-900 mb-2">Prediction Result</h2>
                  <div className="bg-white p-4 rounded border text-center">
                    <div className="text-2xl font-bold text-indigo-600">
                      {predictionResult.prediction_type === 'classification'
                        ? `Predicted Class: ${predictionResult.predicted_value}`
                        : `Predicted Value: ${parseFloat(predictionResult.predicted_value).toFixed(4)}`
                      }
                    </div>
                    {predictionResult.probabilities && (
                      <div className="mt-4">
                        <h3 className="text-md font-medium mb-2">Class Probabilities:</h3>
                        <div className="grid grid-cols-2 gap-2">
                          {Object.entries(predictionResult.probabilities).map(([cls, prob]) => (
                            <div key={cls} className="p-2 bg-gray-50 rounded">
                              <span className="font-medium">Class {cls}:</span> {(prob * 100).toFixed(2)}%
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default PredictiveAnalysis;