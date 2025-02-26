import React, { useState } from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";
import PredictiveAnalysis from "./PredictiveAnalysis";
import DiagnosticAnalysis from "./DiagnosticAnalysis"; // Import new component

// Logo Component
const AnalyzaLogo = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 50" className="h-16">
    <circle cx="30" cy="25" r="12" fill="#4F46E5" />
    <path d="M42 25 L65 25" stroke="#4F46E5" strokeWidth="3" />
    <path d="M42 25 L65 15" stroke="#4F46E5" strokeWidth="3" />
    <path d="M42 25 L65 35" stroke="#4F46E5" strokeWidth="3" />
    <circle cx="70" cy="15" r="5" fill="#818CF8" />
    <circle cx="70" cy="25" r="5" fill="#818CF8" />
    <circle cx="70" cy="35" r="5" fill="#818CF8" />
    <text x="85" y="35" fontFamily="Arial" fontWeight="bold" fontSize="28" fill="#1F2937">
      Analyza
    </text>
  </svg>
  
);

const App = () => {
  const [file, setFile] = useState(null);
  const [query, setQuery] = useState("");
  const [summary, setSummary] = useState("");
  const [graphs, setGraphs] = useState("");
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [analysisType, setAnalysisType] = useState("LLM"); // "LLM" or "Custom"
  const [customAnalysisType, setCustomAnalysisType] = useState(""); // "Predictive" or "Diagnostic"

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { "text/csv": [".csv"] },
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setFile(acceptedFiles[0]);
      }
    },
  });

  const handleSubmit = async () => {
    if (!file || !query) {
      setError("Please upload a CSV file and enter a query.");
      return;
    }

    setError(null);
    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("query", query);

    try {
      const response = await axios.post("http://localhost:8080/analyze", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setSummary(response.data.summary || "No summary available.");
      setGraphs(`data:image/png;base64,${response.data.visualizations}` || "No Visualizations available");
    } catch (error) {
      console.error("Error analyzing data:", error);
      setError("Failed to analyze data. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <AnalyzaLogo />
            <div className="space-x-6">
              <button
                className={`px-4 py-2 text-lg font-semibold ${analysisType === "LLM" ? "text-indigo-600 border-b-2 border-indigo-600" : "text-gray-600"}`}
                onClick={() => setAnalysisType("LLM")}
              >
                LLM Analysis
              </button>
              <button
                className={`px-4 py-2 text-lg font-semibold ${analysisType === "Custom" ? "text-indigo-600 border-b-2 border-indigo-600" : "text-gray-600"}`}
                onClick={() => setAnalysisType("Custom")}
              >
                Custom Analysis
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="bg-white rounded-lg shadow-sm p-8">
          <h2 className="text-3xl font-semibold text-gray-900 mb-8 text-center">
            {analysisType === "LLM" ? "Data Analysis with LLM" : "Custom Data Analysis"}
          </h2>

          {error && (
            <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-md">
              <p className="text-red-600">{error}</p>
            </div>
          )}
          {/* LLM Analysis UI */}

          {analysisType === "LLM" && (
            <>
              {/* File Upload Area */}
              <div {...getRootProps()} className="border-2 border-dashed rounded-lg p-12 text-center hover:border-indigo-500">
                <input {...getInputProps()} />
                <p className="text-gray-600">Upload a CSV file or drag & drop</p>
              </div>

              {file && <p className="text-center mt-2 text-gray-600">Selected file: {file.name}</p>}
              <textarea
                rows="4"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="w-full border rounded-lg p-4 mt-4"
                placeholder="Enter your query..."
              />
              <button
                onClick={handleSubmit}
                className="w-full mt-4 bg-indigo-600 text-white py-3 rounded-md"
              >
                {isLoading ? "Analyzing..." : "Analyze"}
              </button>
              {summary && (
                <div className="mt-8 space-y-4">
                  <h3 className="text-xl font-medium text-gray-900">Analysis Summary</h3>
                  <div className="bg-gray-50 rounded-lg p-6">
                    <p className="text-gray-700 whitespace-pre-wrap">{summary}</p>
                  </div>
                </div>
              )}

              {graphs && <img src={graphs} alt="Generated Visualization" style={{ width: "400px", height: "auto" }} />}
            </>
          )}

          {/* Custom Analysis UI */}
          {analysisType === "Custom" && (
            <>
              <select
                className="w-full border rounded-lg p-4 mt-4"
                value={customAnalysisType}
                onChange={(e) => setCustomAnalysisType(e.target.value)}
              >
                <option value="">Select Analysis Type</option>
                <option value="Predictive">Predictive Analysis</option>
                <option value="Diagnostic">Diagnostic Analysis</option>
              </select>

              {customAnalysisType === "Predictive" && (
                <div className="mt-6">
                  <PredictiveAnalysis />
                </div>
              )}

              {customAnalysisType === "Diagnostic" && (
                <div className="mt-6">
                  <DiagnosticAnalysis />
                </div>
              )}
            </>
          )}
        </div>
      </main>
    </div>
  );
};

export default App;