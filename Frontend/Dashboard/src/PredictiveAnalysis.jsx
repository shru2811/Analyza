import React, { useState, useEffect } from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";

function PredictiveAnalysis() {
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [preview, setPreview] = useState([]);
  const [targetVariable, setTargetVariable] = useState("");
  const [features, setFeatures] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [availableModels, setAvailableModels] = useState([]);
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [prediction, setPrediction] = useState({});
  const [predictionResult, setPredictionResult] = useState(null);
  
  // States for the suggestions feature
  const [useSuggestions, setUseSuggestions] = useState(false);
  const [suggestions, setSuggestions] = useState(null);
  const [targetInfo, setTargetInfo] = useState({ is_categorical: false, is_binary: false });

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
    // Reset states when a new file is uploaded
    setTargetVariable("");
    setFeatures([]);
    setSelectedModel("");
    setResults(null);
    setSuggestions(null);
    setUseSuggestions(false);

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

  // Handle toggle for suggestions
  const handleToggleSuggestions = async (value) => {
    setUseSuggestions(value);
    
    if (value && file) {
      // Request suggestions from the backend
      setIsLoading(true);
      try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("suggest", "true");
        
        const response = await axios.post("http://localhost:8080/predictive-analysis", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        
        console.log("Suggestions:", response.data);
        setSuggestions(response.data);
        
        // If suggestions are available, apply them immediately
        if (response.data.recommended_target) {
          setTargetVariable(response.data.recommended_target);
          
          // Set the recommended features
          if (response.data.recommended_features) {
            setFeatures(response.data.recommended_features);
          }
          
          // Get available models for the recommended target
          await fetchAvailableModels(response.data.recommended_target);
        }
      } catch (error) {
        console.error("Error fetching suggestions:", error);
        setError("Failed to get suggestions. Please try again.");
        setUseSuggestions(false);
      } finally {
        setIsLoading(false);
      }
    } else {
      // If toggling off suggestions, allow manual selection but keep current selections
      // We don't reset values here to maintain the user's current work
    }
  };

  // Effect to apply suggestions when they change or when useSuggestions is toggled
  useEffect(() => {
    if (useSuggestions && suggestions) {
      // Apply suggestions when toggle is turned on and suggestions are available
      if (suggestions.recommended_target) {
        setTargetVariable(suggestions.recommended_target);
      }
      
      if (suggestions.recommended_features) {
        setFeatures(suggestions.recommended_features);
      }
    }
  }, [useSuggestions, suggestions]);

  // Fetch available models based on target type
  const fetchAvailableModels = async (target) => {
    if (!target || !file) return;
    
    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("target_column", target);
      
      const response = await axios.post("http://localhost:8080/get-available-models", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      
      console.log("Available models:", response.data);
      
      if (response.data.available_models?.length > 0) {
        setAvailableModels(response.data.available_models);
        setSelectedModel(response.data.available_models[0]); // Select the first available model
        setTargetInfo(response.data.target_info);
      } else {
        setAvailableModels([]);
        setSelectedModel("");
      }
    } catch (error) {
      console.error("Error fetching available models:", error);
      setError("Failed to determine available models for the selected target.");
    } finally {
      setIsLoading(false);
    }
  };

  // When target variable changes, fetch available models
  useEffect(() => {
    if (targetVariable) {
      fetchAvailableModels(targetVariable);
    } else {
      setAvailableModels([]);
      setSelectedModel("");
    }
  }, [targetVariable]);

  const handleAnalyze = async () => {
    if (!targetVariable || features.length === 0 || !selectedModel) {
      setError("Please select a target variable, at least one feature, and a model.");
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
      formData.append("suggest", "false");

      const response = await axios.post("http://localhost:8080/predictive-analysis", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      console.log("Analysis results:", response.data);
      setResults(response.data);
      
      // Update target info if available in response
      if (response.data.target_info) {
        setTargetInfo(response.data.target_info);
      }

      // Initialize prediction state with feature fields
      const initialPrediction = {};
      features.forEach(feature => {
        initialPrediction[feature] = "";
      });
      setPrediction(initialPrediction);
      setPredictionResult(null);
    } catch (error) {
      console.error("Error analyzing data:", error);
      setError(error.response?.data?.detail || "Failed to analyze data. Please try again.");
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

  // Helper function to render data preview table
  const renderDataPreview = () => {
    if (!preview || preview.length === 0) {
      return <p className="text-gray-600">No preview data available.</p>;
    }

    return (
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse">
          <thead>
            <tr className="bg-gray-100">
              {columns.map(column => (
                <th key={column} className="border p-2 text-left font-medium text-sm">
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.map((row, rowIndex) => (
              <tr key={rowIndex} className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                {columns.map(column => (
                  <td key={`${rowIndex}-${column}`} className="border p-2 text-sm">
                    {row[column] !== undefined ? 
                      (typeof row[column] === 'number' ? 
                        row[column].toFixed(2) : 
                        String(row[column])) : 
                      'N/A'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // Render column type information from suggestions
  const renderColumnTypeInfo = (column) => {
    if (!suggestions?.column_types || !suggestions.column_types[column]) return null;
    
    const colInfo = suggestions.column_types[column];
    return (
      <span className="text-xs text-gray-500 ml-2">
        ({colInfo.type}{colInfo.is_binary ? ", binary" : ""}, {colInfo.unique_values} unique values)
      </span>
    );
  };

  return (
    <div className="space-y-6">
      {/* File Upload Area */}
      <div
        {...getRootProps()}
        className="border-2 border-dashed rounded-lg p-6 text-center hover:border-indigo-500 cursor-pointer"
      >
        <input {...getInputProps()} />
        <p className="text-gray-600">{file ? `Selected: ${file.name}` : "Upload a CSV file or drag & drop (Upto 200MB)"}</p>
        <p className="text-sm text-gray-500">
            Supports .csv files
          </p>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-md">
          <p className="text-red-600">{error}</p>
        </div>
      )}

      {isLoading && <p className="text-center text-gray-600">Loading...</p>}

      {columns.length > 0 && (
        <div className="space-y-6">
          {/* Data Preview Section */}
          <div>
            <h2 className="text-lg font-medium text-gray-900 mb-2">Data Preview</h2>
            <div className="bg-gray-50 p-3 rounded-lg">
              {renderDataPreview()}
            </div>
            <p className="text-xs text-gray-500 mt-1">Showing up to 5 rows of data</p>
          </div>
          
          {/* Suggestions Toggle */}
          <div className="flex items-center">
            <label className="mr-2 text-gray-700">Use suggested target and features?</label>
            <label className="relative inline-flex items-center cursor-pointer">
              <input 
                type="checkbox" 
                className="sr-only peer"
                checked={useSuggestions}
                onChange={(e) => handleToggleSuggestions(e.target.checked)}
              />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-indigo-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-600"></div>
            </label>
          </div>

          <div>
            <h2 className="text-lg font-medium text-gray-900 mb-2">Select Target Variable</h2>
            <select
              className="w-full border rounded-lg p-2"
              value={targetVariable}
              onChange={(e) => setTargetVariable(e.target.value)}
              disabled={useSuggestions && suggestions?.recommended_target}
            >
              <option value="">Select</option>
              {columns.map((col) => (
                <option key={col} value={col}>
                  {col} {renderColumnTypeInfo(col)}
                </option>
              ))}
            </select>
            {suggestions?.recommended_target && useSuggestions && (
              <p className="mt-2 text-sm text-indigo-600">
                Recommended target: {suggestions.recommended_target} 
                {suggestions.column_types[suggestions.recommended_target] && 
                  ` (${suggestions.column_types[suggestions.recommended_target].type})`}
              </p>
            )}
          </div>

          <div>
            <h2 className="text-lg font-medium text-gray-900 mb-2">Select Features</h2>
            <select
              multiple
              className="w-full border rounded-lg p-2 h-32"
              onChange={(e) => setFeatures([...e.target.selectedOptions].map(o => o.value))}
              value={features}
              disabled={useSuggestions && suggestions?.recommended_features}
            >
              {columns.filter(col => col !== targetVariable).map((col) => (
                <option key={col} value={col}>
                  {col} {renderColumnTypeInfo(col)}
                </option>
              ))}
            </select>
            {features.length > 0 && (
              <p className="mt-2 text-sm text-gray-600">
                Selected features: {features.join(", ")}
              </p>
            )}
            {suggestions?.recommended_features && useSuggestions && (
              <p className="mt-2 text-sm text-indigo-600">
                Recommended features: {suggestions.recommended_features.join(", ")}
              </p>
            )}
          </div>

          <div>
            <h2 className="text-lg font-medium text-gray-900 mb-2">Select Model</h2>
            <select
              className="w-full border rounded-lg p-2"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={availableModels.length === 0}
            >
              <option value="">Select</option>
              {availableModels.map((model) => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
            {targetInfo.is_categorical && (
              <p className="mt-2 text-sm text-gray-600">
                {targetInfo.is_binary 
                  ? "The target is binary categorical - both Logistic Regression and Random Forest are available."
                  : "The target is multi-class categorical - only Random Forest is available."}
              </p>
            )}
            {!targetInfo.is_categorical && targetVariable && (
              <p className="mt-2 text-sm text-gray-600">
                The target is numeric - both Linear Regression and Random Forest are available.
              </p>
            )}
          </div>

          <button
            onClick={handleAnalyze}
            className="w-full bg-indigo-600 text-white p-3 rounded-md hover:bg-indigo-700 transition disabled:bg-indigo-300"
            disabled={isLoading || !targetVariable || features.length === 0 || !selectedModel}
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
                  {/* {results.confusion_matrix && (
                    <div className="mt-4">
                      <h3 className="text-md font-medium">Confusion Matrix:</h3>
                      <div className="overflow-x-auto mt-2">
                        <table className="min-w-full border-collapse">
                          <tbody>
                            {results.confusion_matrix.map((row, i) => (
                              <tr key={i}>
                                {row.map((cell, j) => (
                                  <td key={j} className="border p-2 text-center">
                                    {cell}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )} */}
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
                    className="w-full bg-indigo-600 text-white p-3 rounded-md hover:bg-indigo-700 transition mt-4 disabled:bg-indigo-300"
                    disabled={isLoading || features.length === 0}
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