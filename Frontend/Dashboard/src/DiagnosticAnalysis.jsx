import React, { useState } from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";

function DiagnosticAnalysis() {
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [preview, setPreview] = useState([]);
  const [targetVariable, setTargetVariable] = useState("");
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [correlationMatrix, setCorrelationMatrix] = useState(null);
  const [anomalies, setAnomalies] = useState([]);
  const [rootCauseAnalysis, setRootCauseAnalysis] = useState(null);

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

  const handleDiagnosticAnalysis = async () => {
    if (!file) {
      setError("Please upload a file first.");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      
      // If a target variable is selected, include it in the request
      if (targetVariable) {
        formData.append("target_variable", targetVariable);
      }

      const response = await axios.post("http://localhost:8080/diagnostic-analysis", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      
      // Process and store the results
      setResults(response.data);
      setCorrelationMatrix(response.data.correlation_matrix);
      setAnomalies(response.data.anomalies || []);
      setRootCauseAnalysis(response.data.root_cause_analysis);
    } catch (error) {
      console.error("Error performing diagnostic analysis:", error);
      setError("Failed to perform diagnostic analysis. Please try again.");
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

  // Helper function to render correlation matrix as a heatmap
  const renderCorrelationMatrix = () => {
    if (!correlationMatrix) return null;
    
    const variables = Object.keys(correlationMatrix);
    
    return (
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse">
          <thead>
            <tr>
              <th className="border p-2 bg-gray-100"></th>
              {variables.map(variable => (
                <th key={variable} className="border p-2 bg-gray-100 text-sm">
                  {variable}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {variables.map(rowVar => (
              <tr key={rowVar}>
                <th className="border p-2 bg-gray-100 text-sm text-left">{rowVar}</th>
                {variables.map(colVar => {
                  const correlation = correlationMatrix[rowVar][colVar];
                  const intensity = Math.abs(correlation);
                  const color = correlation > 0 
                    ? `rgba(0, 0, 255, ${intensity})` 
                    : `rgba(255, 0, 0, ${intensity})`;
                  
                  return (
                    <td 
                      key={colVar} 
                      className="border p-2 text-center text-sm"
                      style={{ backgroundColor: color, color: intensity > 0.5 ? 'white' : 'black' }}
                    >
                      {correlation?.toFixed(2) || 'N/A'}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // Helper function to render root cause analysis
  const renderRootCauseAnalysis = () => {
    if (!rootCauseAnalysis || Object.keys(rootCauseAnalysis).length === 0) {
      return <p className="text-gray-600">No root cause analysis available. Select a target variable.</p>;
    }
    
    const factors = Object.entries(rootCauseAnalysis)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    
    return (
      <div className="space-y-2">
        <h3 className="font-medium">Factors influencing {targetVariable}:</h3>
        <ul className="space-y-1">
          {factors.map(([factor, correlation]) => (
            <li key={factor} className="flex items-center">
              <div 
                className="w-1/2 h-4 bg-gray-200 rounded-full overflow-hidden"
              >
                <div 
                  className={`h-full ${correlation > 0 ? 'bg-blue-500' : 'bg-red-500'}`}
                  style={{ width: `${Math.abs(correlation) * 100}%` }}
                ></div>
              </div>
              <span className="ml-2">
                {factor}: {correlation.toFixed(2)}
              </span>
            </li>
          ))}
        </ul>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <p className="pb-4 px-8 text-center">Leverage the power of AI to explore your data through natural language. Our LLM-powered analysis uses Google's Gemini 1.5 to generate instant insights, detect patterns, and answer complex questions about your dataset - no coding required. Simply upload your data and ask questions in plain English.</p>

      {/* File Upload Area */}
      <div 
        {...getRootProps()} 
        className="border-2 border-dashed rounded-lg p-6 text-center hover:border-indigo-500 cursor-pointer"
      >
        <input {...getInputProps()} />
        <p className="text-gray-600">{file ? `Selected: ${file.name}` : "Upload a CSV file or drag & drop (upto 200MB)"}</p>
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

          {/* Optional Target Variable Selection for Root Cause Analysis */}
          <div>
            <h2 className="text-lg font-medium text-gray-900 mb-2">
              Select Target Variable (Optional)
            </h2>
            <p className="text-sm text-gray-600 mb-2">
              Selecting a target variable will enable root cause analysis
            </p>
            <select 
              className="w-full border rounded-lg p-2"
              value={targetVariable} 
              onChange={(e) => setTargetVariable(e.target.value)}
            >
              <option value="">None (General Diagnostics)</option>
              {columns.map((col) => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
          </div>

          <button 
            onClick={handleDiagnosticAnalysis} 
            className="w-full bg-indigo-600 text-white p-3 rounded-md hover:bg-indigo-700 transition"
            disabled={isLoading}
          >
            {isLoading ? "Analyzing..." : "Run Diagnostic Analysis"}
          </button>

          {results && (
            <div className="space-y-6">
              {/* Correlation Matrix */}
              <div className="bg-white p-4 rounded-lg border">
                <h2 className="text-lg font-medium text-gray-900 mb-4">Correlation Matrix</h2>
                {renderCorrelationMatrix()}
              </div>
              
              {/* Anomalies */}
              <div className="bg-white p-4 rounded-lg border">
                <h2 className="text-lg font-medium text-gray-900 mb-4">Anomalies Detected</h2>
                {anomalies.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="min-w-full border-collapse">
                      <thead>
                        <tr>
                          {Object.keys(anomalies[0] || {}).map(key => (
                            <th key={key} className="border p-2 bg-gray-100">{key}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {anomalies.map((anomaly, index) => (
                          <tr key={index} className="bg-red-50">
                            {Object.values(anomaly).map((value, i) => (
                              <td key={i} className="border p-2">{typeof value === 'number' ? value.toFixed(2) : value}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="text-gray-600">No anomalies detected in the dataset.</p>
                )}
              </div>
              
              {/* Root Cause Analysis */}
              <div className="bg-white p-4 rounded-lg border">
                <h2 className="text-lg font-medium text-gray-900 mb-4">Root Cause Analysis</h2>
                {renderRootCauseAnalysis()}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default DiagnosticAnalysis;