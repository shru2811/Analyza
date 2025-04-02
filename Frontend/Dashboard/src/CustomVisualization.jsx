import React, { useState } from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";

const CustomVisualization = () => {
  const [file, setFile] = useState(null);
  const [dataPreview, setDataPreview] = useState([]);
  const [columns, setColumns] = useState([]);
  const [visualizations, setVisualizations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Configure dropzone for file uploads
  const { getRootProps, getInputProps } = useDropzone({
    accept: { "text/csv": [".csv"] },
    onDrop: async (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        await handleFileChange(acceptedFiles[0]);
      }
    },
  });

  // Handle file upload
  const handleFileChange = async (uploadedFile) => {
    if (!uploadedFile) {
      setError("Please select a file first.");
      return;
    }

    setFile(uploadedFile);
    setIsLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append("file", uploadedFile);

    try {
      // Check the correct API endpoint URL - ensure it matches your backend
      const response = await axios.post("http://localhost:8080/preview-data/", formData, {
        headers: { 
          "Content-Type": "multipart/form-data" 
        },
        // Add timeout to prevent hanging requests
        timeout: 30000
      });
      
      // Debug the response
      console.log("Preview data response:", response.data);
      
      if (response.data && response.data.preview) {
        setDataPreview(response.data.preview || []);
        setColumns(response.data.columns || []);
      } else {
        setError("Invalid response format from the server. Check the API response structure.");
        console.error("Invalid response format:", response.data);
      }
    } catch (error) {
      console.error("Error fetching data preview:", error);
      
      // More detailed error handling
      if (error.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        setError(`Server error: ${error.response.status} - ${error.response.data.message || 'Unknown error'}`);
        console.error("Server response:", error.response.data);
      } else if (error.request) {
        // The request was made but no response was received
        setError("No response from server. Please check if the backend service is running.");
      } else {
        // Something happened in setting up the request that triggered an Error
        setError(`Error: ${error.message}`);
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Handle adding a new visualization
  const handleAddVisualization = async (graphType) => {
    if (!file) {
      setError("Please upload a file first.");
      return;
    }

    // Add the graph immediately with default values
    setVisualizations([
      ...visualizations,
      {
        type: graphType,
        image: null, // Initially no image
        xAxis: "",
        yAxis: [],
        colors: ["#1f77b4"], // Default color
        explode: 0.0, // For pie chart
        bins: 20, // For histogram
      },
    ]);
  };

  // Handle updating axes and colors for a specific graph
  const handleUpdateGraph = async (index, xAxis, yAxis, colors, explode, bins) => {
    if (!file) {
      setError("Please upload a file first.");
      return;
    }

    // Validate inputs before making the request
    if (visualizations[index].type !== "histogram" && visualizations[index].type !== "pie" && !xAxis) {
      setError("Please select an X-Axis value");
      return;
    }
    
    if (yAxis.length === 0) {
      setError("Please select at least one Y-Axis value");
      return;
    }

    setIsLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", file);

    try {
      // Build the URL with proper encoding for all parameters
      const params = new URLSearchParams();
      params.append("plot_type", visualizations[index].type);
      if (xAxis) params.append("x_axis", xAxis);
      params.append("y_axis", yAxis.join(","));
      params.append("colors", colors.join(","));
      params.append("explode", explode);
      params.append("bins", bins);
      
      const url = `http://localhost:8080/custom-visualization/?${params.toString()}`;
      console.log("Request URL:", url);
      
      const response = await axios.post(url, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 60000 // Increase timeout for visualization generation
      });

      console.log("Visualization response:", response.data);

      if (response.data && response.data.image) {
        const updatedVisualizations = [...visualizations];
        updatedVisualizations[index].image = response.data.image;
        updatedVisualizations[index].xAxis = xAxis;
        updatedVisualizations[index].yAxis = yAxis;
        updatedVisualizations[index].colors = colors;
        updatedVisualizations[index].explode = explode;
        updatedVisualizations[index].bins = bins;
        setVisualizations(updatedVisualizations);
      } else {
        setError(response.data.error || "No visualization generated. Check your selections.");
      }
    } catch (error) {
      console.error("Error updating visualization:", error);
      
      // Detailed error handling
      if (error.response) {
        setError(`Server error: ${error.response.status} - ${error.response.data.message || 'Error generating visualization'}`);
      } else if (error.request) {
        setError("No response from server. Please check if the backend service is running.");
      } else {
        setError(`Error: ${error.message}`);
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Handle removing a visualization
  const handleRemoveVisualization = (index) => {
    const updatedVisualizations = visualizations.filter((_, i) => i !== index);
    setVisualizations(updatedVisualizations);
  };

  // Function to retry loading with the current file
  const handleRetry = () => {
    if (file) {
      handleFileChange(file);
    }
  };

  return (
    <div className="min-h-screen  p-5">
      <p className="pb-4 px-8 text-center">Transform your data into meaningful visuals. Choose from multiple chart types, customize axes, apply filters, and highlight key trends. All visualizations are interactive - hover for details, zoom into areas of interest, or export for presentations</p>

      {/* File Upload Section with Dropzone */}
      <div className="">
        {/* <h2 className="text-lg font-semibold mb-4 text-center">Upload Your Data</h2> */}
        
        <div 
          {...getRootProps()} 
          className="border-2 border-dashed rounded-lg p-6 text-center hover:border-blue-500 cursor-pointer transition-colors"
        >
          <input {...getInputProps()} />
          <p className="text-gray-600 mb-2">
            {file ? `Selected: ${file.name}` : "Upload a CSV file or drag & drop (Upto 200MB)"}
          </p>
          <p className="text-sm text-gray-500">
            Supports .csv files
          </p>
        </div>
        
        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
            <p className="text-red-600 text-sm">{error}</p>
            <button 
              onClick={handleRetry}
              className="mt-2 text-sm text-blue-600 hover:text-blue-800"
            >
              Try again
            </button>
          </div>
        )}
        
        {isLoading && (
          <div className="mt-4 text-center">
            <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-blue-600 border-r-transparent align-middle"></div>
            <p className="mt-2 text-gray-600">Processing data...</p>
          </div>
        )}
      </div>

      {/* Data Preview Section */}
      {dataPreview.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow-md max-w-3xl mx-auto mt-4">
          <h2 className="text-lg font-semibold mb-2">Data Preview</h2>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse border border-gray-300 text-sm">
              <thead>
                <tr className="bg-gray-100">
                  {columns.map((col, idx) => (
                    <th key={idx} className="border border-gray-300 p-2 text-left">{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dataPreview.map((row, rowIdx) => (
                  <tr key={rowIdx} className={rowIdx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    {columns.map((col, colIdx) => (
                      <td key={colIdx} className="border border-gray-300 p-2">
                        {row[col] !== undefined ? 
                          (typeof row[col] === 'number' ? 
                            row[col].toFixed(2) : 
                            String(row[col])) : 
                          'N/A'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-500 mt-2">Showing first {dataPreview.length} rows of data</p>
        </div>
      )}

      {/* Graph Selection Section */}
      {columns.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow-md max-w-3xl mx-auto mt-4">
          <h2 className="text-lg font-semibold mb-2">Select Graph Type</h2>
          <select
            onChange={(e) => e.target.value && handleAddVisualization(e.target.value)}
            className="p-2 border rounded w-full"
            value=""
            disabled={isLoading}
          >
            <option value="">Select Graph Type</option>
            <option value="bar">Bar Chart</option>
            <option value="scatter">Scatter Plot</option>
            <option value="line">Line Graph</option>
            <option value="histogram">Histogram</option>
            <option value="box">Box Plot</option>
            <option value="violin">Violin Plot</option>
            <option value="pie">Pie Chart</option>
          </select>
        </div>
      )}

      {/* Visualization Dashboard */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
        {visualizations.map((viz, index) => (
          <div key={index} className="bg-white p-4 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-2">{viz.type.charAt(0).toUpperCase() + viz.type.slice(1)} Plot</h3>
            {viz.image ? (
              <img src={`data:image/png;base64,${viz.image}`} alt={`Visualization ${index + 1}`} className="mx-auto" />
            ) : (
              <div className="text-center py-10 bg-gray-50 rounded border border-gray-200">
                <p className="text-gray-500">No graph generated yet.</p>
                <p className="text-sm text-gray-400 mt-1">Configure and update to generate</p>
              </div>
            )}
            <GraphControls
              columns={columns}
              xAxis={viz.xAxis}
              yAxis={viz.yAxis}
              colors={viz.colors}
              graphType={viz.type}
              explode={viz.explode}
              bins={viz.bins}
              onUpdate={(xAxis, yAxis, colors, explode, bins) => handleUpdateGraph(index, xAxis, yAxis, colors, explode, bins)}
              isLoading={isLoading}
            />
            <button
              onClick={() => handleRemoveVisualization(index)}
              className="mt-2 bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600 w-full"
              disabled={isLoading}
            >
              Remove
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

// Graph Controls Component
const GraphControls = ({ columns, xAxis, yAxis, colors, graphType, explode, bins, onUpdate, isLoading }) => {
  const [selectedXAxis, setSelectedXAxis] = useState(xAxis);
  const [selectedYAxis, setSelectedYAxis] = useState(yAxis);
  const [selectedColors, setSelectedColors] = useState(colors);
  const [selectedExplode, setSelectedExplode] = useState(explode);
  const [selectedBins, setSelectedBins] = useState(bins);

  // Reset to props when graphType changes
  React.useEffect(() => {
    setSelectedXAxis(xAxis);
    setSelectedYAxis(yAxis);
    setSelectedColors(colors);
    setSelectedExplode(explode);
    setSelectedBins(bins);
  }, [graphType, xAxis, yAxis, colors, explode, bins]);

  const handleUpdate = () => {
    onUpdate(selectedXAxis, selectedYAxis, selectedColors, selectedExplode, selectedBins);
  };

  // Filter columns by data type
  const getNumericColumns = () => {
    // If we had type information, we'd filter here
    // For now we're just returning all columns
    return columns;
  };

  return (
    <div className="mt-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
      {/* Hide X-Axis for histogram and pie chart */}
      {graphType !== "histogram" && graphType !== "pie" && (
        <div className="mb-4">
          <label className="block text-sm font-medium mb-1">X-Axis</label>
          <select
            value={selectedXAxis}
            onChange={(e) => setSelectedXAxis(e.target.value)}
            className="p-2 border rounded w-full"
            disabled={isLoading}
          >
            <option value="">Select X-Axis</option>
            {columns.map((col, idx) => (
              <option key={idx} value={col}>{col}</option>
            ))}
          </select>
        </div>
      )}

      <div className="mb-4">
        <label className="block text-sm font-medium mb-1">
          {graphType === "pie" ? "Values" : "Y-Axis"} 
          {graphType !== "pie" && <span className="text-xs text-gray-500 ml-1">(Hold Ctrl/Cmd to select multiple)</span>}
        </label>
        <select
          multiple={graphType !== "pie"}
          value={selectedYAxis}
          onChange={(e) => setSelectedYAxis(graphType !== "pie" ? 
            [...e.target.selectedOptions].map((opt) => opt.value) : 
            [e.target.value])}
          className="p-2 border rounded w-full"
          disabled={isLoading}
          size={graphType !== "pie" ? Math.min(4, columns.length) : 1}
        >
          {(graphType === "histogram" ? getNumericColumns() : columns).map((col, idx) => (
            <option key={idx} value={col}>{col}</option>
          ))}
        </select>
        {graphType === "histogram" && (
          <p className="text-xs text-gray-500 mt-1">Only numeric columns can be used for histograms</p>
        )}
      </div>

      {/* Additional controls for pie chart and histogram */}
      {graphType === "pie" && (
        <div className="mb-4">
          <label className="block text-sm font-medium mb-1">
            Explode Slice: {selectedExplode.toFixed(2)}
          </label>
          <input
            type="range"
            min="0"
            max="0.5"
            step="0.05"
            value={selectedExplode}
            onChange={(e) => setSelectedExplode(parseFloat(e.target.value))}
            className="w-full"
            disabled={isLoading}
          />
        </div>
      )}

      {graphType === "histogram" && (
        <div className="mb-4">
          <label className="block text-sm font-medium mb-1">Number of Bins</label>
          <input
            type="number"
            min="5"
            max="100"
            value={selectedBins}
            onChange={(e) => setSelectedBins(parseInt(e.target.value))}
            className="p-2 border rounded w-full"
            disabled={isLoading}
          />
        </div>
      )}

      <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Colors</label>
        <div className="flex flex-wrap gap-2">
          {(selectedYAxis.length > 0 ? selectedYAxis : ['']).map((_, index) => (
            <input
              key={index}
              type="color"
              value={selectedColors[index] || "#1f77b4"}
              onChange={(e) => {
                const newColors = [...selectedColors];
                newColors[index] = e.target.value;
                setSelectedColors(newColors);
              }}
              className="p-0 h-8 w-8 border rounded cursor-pointer"
              disabled={isLoading}
            />
          ))}
        </div>
      </div>

      <button
        onClick={handleUpdate}
        className="w-full bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:bg-blue-300"
        disabled={isLoading}
      >
        {isLoading ? 
          <span className="flex items-center justify-center">
            <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Updating...
          </span> : 
          "Update Graph"
        }
      </button>
    </div>
  );
};

export default CustomVisualization;