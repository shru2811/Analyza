import React, { useState } from "react";
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
      
      setResults(response.data);
    } catch (error) {
      console.error("Error analyzing data:", error);
      setError("Failed to analyze data. Please try again.");
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
            <div className="bg-gray-50 p-4 rounded-lg">
              <h2 className="text-lg font-medium text-gray-900 mb-2">Results</h2>
              <pre className="bg-white p-4 rounded border overflow-x-auto">
                {JSON.stringify(results, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default PredictiveAnalysis;