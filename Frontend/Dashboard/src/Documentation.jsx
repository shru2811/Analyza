import { useState } from 'react';

const Documentation = () => {
  const [activeSection, setActiveSection] = useState('overview');

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-blue-600 text-white p-6 shadow-md">
        <div className="container mx-auto">
          <h1 className="text-3xl font-bold">Analyza Documentation</h1>
          <p className="mt-2">Data Analysis & Visualization Tool</p>
        </div>
      </header>

      <div className="container mx-auto p-6 flex flex-col md:flex-row gap-8">
        {/* Sidebar Navigation */}
        <aside className="w-full md:w-64 bg-white p-6 rounded-lg shadow-md h-fit sticky top-6">
          <nav>
            <ul className="space-y-2">
              <li>
                <button 
                  onClick={() => setActiveSection('overview')}
                  className={`w-full text-left p-2 rounded ${activeSection === 'overview' ? 'bg-blue-100 text-blue-600' : 'hover:bg-gray-100'}`}
                >
                  Overview
                </button>
              </li>
              <li>
                <button 
                  onClick={() => setActiveSection('features')}
                  className={`w-full text-left p-2 rounded ${activeSection === 'features' ? 'bg-blue-100 text-blue-600' : 'hover:bg-gray-100'}`}
                >
                  Key Features
                </button>
              </li>
              <li>
                <button 
                  onClick={() => setActiveSection('modules')}
                  className={`w-full text-left p-2 rounded ${activeSection === 'modules' ? 'bg-blue-100 text-blue-600' : 'hover:bg-gray-100'}`}
                >
                  Modules
                </button>
              </li>
              <li>
                <button 
                  onClick={() => setActiveSection('techstack')}
                  className={`w-full text-left p-2 rounded ${activeSection === 'techstack' ? 'bg-blue-100 text-blue-600' : 'hover:bg-gray-100'}`}
                >
                  Technology Stack
                </button>
              </li>
              <li>
                <button 
                  onClick={() => setActiveSection('userguide')}
                  className={`w-full text-left p-2 rounded ${activeSection === 'userguide' ? 'bg-blue-100 text-blue-600' : 'hover:bg-gray-100'}`}
                >
                  User Guide
                </button>
              </li>
              <li>
                <button 
                  onClick={() => setActiveSection('limitations')}
                  className={`w-full text-left p-2 rounded ${activeSection === 'limitations' ? 'bg-blue-100 text-blue-600' : 'hover:bg-gray-100'}`}
                >
                  Limitations
                </button>
              </li>
            </ul>
          </nav>
        </aside>

        {/* Main Content */}
        <main className="flex-1 bg-white p-8 rounded-lg shadow-md">
          {/* Overview Section */}
          {activeSection === 'overview' && (
            <section>
              <h2 className="text-2xl font-bold mb-4 text-blue-700">Overview</h2>
              <div className="space-y-6">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h3 className="text-xl font-semibold mb-2">What is Analyza?</h3>
                  <p className="text-gray-700">
                    Analyza is a web-based data analysis and visualization tool designed to simplify complex data processing and enhance decision-making. 
                    It integrates machine learning (ML) techniques for predictive and diagnostic analysis, while leveraging large language models (LLMs) 
                    for descriptive analysis, enabling users to gain valuable insights from their data.
                  </p>
                </div>

                <div>
                  <h3 className="text-xl font-semibold mb-2">Purpose</h3>
                  <p className="text-gray-700 mb-4">
                    Analyza aims to bridge the gap between raw data and actionable insights by providing an intuitive interface for users to explore and analyze 
                    data without requiring programming expertise.
                  </p>
                  <ul className="list-disc pl-6 space-y-2 text-gray-700">
                    <li>Simplify data analysis for non-technical users</li>
                    <li>Provide advanced analytical capabilities for data professionals</li>
                    <li>Enable data-driven decision making across various domains</li>
                    <li>Offer scalable batch processing for large datasets</li>
                  </ul>
                </div>

                <div>
                  <h3 className="text-xl font-semibold mb-2">Target Users</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h4 className="font-medium text-blue-600">Business Professionals</h4>
                      <p className="text-gray-700 text-sm">Market trend analysis, customer behavior insights</p>
                    </div>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h4 className="font-medium text-blue-600">Researchers</h4>
                      <p className="text-gray-700 text-sm">Pattern identification, hypothesis validation</p>
                    </div>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h4 className="font-medium text-blue-600">Financial Experts</h4>
                      <p className="text-gray-700 text-sm">Risk assessment, market trend prediction</p>
                    </div>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h4 className="font-medium text-blue-600">Educators & Students</h4>
                      <p className="text-gray-700 text-sm">Academic research, learning data analysis</p>
                    </div>
                  </div>
                </div>
              </div>
            </section>
          )}

          {/* Features Section */}
          {activeSection === 'features' && (
            <section>
              <h2 className="text-2xl font-bold mb-6 text-blue-700">Key Features</h2>
              <div className="space-y-6">
                <div className="border-l-4 border-blue-500 pl-4">
                  <h3 className="text-xl font-semibold mb-2">Interactive Dashboards</h3>
                  <p className="text-gray-700">
                    Dynamic, customizable visualizations including bar charts, histograms, scatter plots, and heatmaps. 
                    Users can filter, drill down, and compare different data points interactively.
                  </p>
                </div>

                <div className="border-l-4 border-blue-500 pl-4">
                  <h3 className="text-xl font-semibold mb-2">Machine Learning Integration</h3>
                  <p className="text-gray-700">
                    Incorporates descriptive, predictive, and diagnostic analytics using machine learning models:
                  </p>
                  <ul className="list-disc pl-6 mt-2 space-y-1 text-gray-700">
                    <li><span className="font-medium">Descriptive:</span> LLM-powered summaries and insights</li>
                    <li><span className="font-medium">Predictive:</span> Linear Regression, Logistic Regression, Random Forest</li>
                    <li><span className="font-medium">Diagnostic:</span> Correlation analysis, anomaly detection, root cause analysis</li>
                  </ul>
                </div>

                <div className="border-l-4 border-blue-500 pl-4">
                  <h3 className="text-xl font-semibold mb-2">Custom Visualization</h3>
                  <p className="text-gray-700">
                    Generate tailored visualizations based on your data with options for:
                  </p>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
                    <span className="bg-gray-100 px-3 py-1 rounded-full text-sm text-center">Bar Charts</span>
                    <span className="bg-gray-100 px-3 py-1 rounded-full text-sm text-center">Scatter Plots</span>
                    <span className="bg-gray-100 px-3 py-1 rounded-full text-sm text-center">Line Graphs</span>
                    <span className="bg-gray-100 px-3 py-1 rounded-full text-sm text-center">Histograms</span>
                    <span className="bg-gray-100 px-3 py-1 rounded-full text-sm text-center">Pie Charts</span>
                    <span className="bg-gray-100 px-3 py-1 rounded-full text-sm text-center">Box Plots</span>
                    <span className="bg-gray-100 px-3 py-1 rounded-full text-sm text-center">Violin Plots</span>
                    <span className="bg-gray-100 px-3 py-1 rounded-full text-sm text-center">Heatmaps</span>
                  </div>
                </div>

                <div className="border-l-4 border-blue-500 pl-4">
                  <h3 className="text-xl font-semibold mb-2">Data Pre-processing</h3>
                  <p className="text-gray-700">
                    Automated handling of:
                  </p>
                  <ul className="list-disc pl-6 mt-2 space-y-1 text-gray-700">
                    <li>Missing values</li>
                    <li>Outlier detection and correction</li>
                    <li>Data standardization</li>
                    <li>Categorical variable encoding</li>
                  </ul>
                </div>

                <div className="border-l-4 border-blue-500 pl-4">
                  <h3 className="text-xl font-semibold mb-2">User-Friendly Interface</h3>
                  <p className="text-gray-700">
                    Intuitive, guided workflow with interactive tooltips, predefined templates, 
                    and AI-powered recommendations to assist users at every stage of data exploration.
                  </p>
                </div>
              </div>
            </section>
          )}

          {/* Modules Section */}
          {activeSection === 'modules' && (
            <section>
              <h2 className="text-2xl font-bold mb-6 text-blue-700">Modules</h2>
              <div className="space-y-8">
                <div className="bg-gray-50 p-6 rounded-lg">
                  <h3 className="text-xl font-semibold mb-3 text-blue-600">LLM Analysis Module</h3>
                  <p className="text-gray-700 mb-4">
                    Uses Gemini Flash 1.5 Model to generate automated insights and summaries from user-uploaded datasets. 
                    Provides a natural language interface for users to query their data.
                  </p>
                  <div className="bg-white p-4 rounded border">
                    <h4 className="font-medium mb-2">Key Functionalities:</h4>
                    <ul className="list-disc pl-6 space-y-1 text-gray-700">
                      <li>Natural language query processing</li>
                      <li>Automated dataset summarization</li>
                      <li>Pattern and trend identification</li>
                      <li>Visualization generation based on queries</li>
                    </ul>
                  </div>
                </div>

                <div className="bg-gray-50 p-6 rounded-lg">
                  <h3 className="text-xl font-semibold mb-3 text-blue-600">Custom Analysis Module</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-white p-4 rounded border">
                      <h4 className="font-medium mb-2 text-blue-500">Predictive Analysis</h4>
                      <ul className="list-disc pl-6 space-y-1 text-gray-700">
                        <li><span className="font-medium">Linear Regression:</span> For continuous value prediction</li>
                        <li><span className="font-medium">Logistic Regression:</span> For binary classification</li>
                        <li><span className="font-medium">Random Forest:</span> For both classification and regression</li>
                        <li>Feature selection interface</li>
                        <li>Performance metrics display</li>
                      </ul>
                    </div>
                    <div className="bg-white p-4 rounded border">
                      <h4 className="font-medium mb-2 text-blue-500">Diagnostic Analysis</h4>
                      <ul className="list-disc pl-6 space-y-1 text-gray-700">
                        <li>Correlation matrix generation</li>
                        <li>Anomaly detection using Z-score method</li>
                        <li>Root cause analysis</li>
                        <li>Interactive heatmaps for relationship visualization</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 p-6 rounded-lg">
                  <h3 className="text-xl font-semibold mb-3 text-blue-600">Custom Visualization Module</h3>
                  <p className="text-gray-700 mb-4">
                    Enables users to create interactive charts, customize axes and colors, and explore data patterns effortlessly.
                  </p>
                  <div className="bg-white p-4 rounded border">
                    <h4 className="font-medium mb-2">Features:</h4>
                    <ul className="list-disc pl-6 space-y-1 text-gray-700">
                      <li>Multiple chart type selection</li>
                      <li>Axis customization (X, Y variables)</li>
                      <li>Color coding options</li>
                      <li>Interactive chart elements</li>
                      <li>Export options for visualizations</li>
                    </ul>
                  </div>
                </div>
              </div>
            </section>
          )}

          {/* Tech Stack Section */}
          {activeSection === 'techstack' && (
            <section>
              <h2 className="text-2xl font-bold mb-6 text-blue-700">Technology Stack</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gray-50 p-6 rounded-lg">
                  <h3 className="text-xl font-semibold mb-3 text-blue-600">Frontend</h3>
                  <ul className="space-y-3">
                    <li className="flex items-start">
                      <span className="bg-blue-100 text-blue-800 p-2 rounded mr-3">React.js</span>
                      <span className="text-gray-700">JavaScript library for building user interfaces</span>
                    </li>
                    <li className="flex items-start">
                      <span className="bg-blue-100 text-blue-800 p-2 rounded mr-3">Vite</span>
                      <span className="text-gray-700">Next generation frontend tooling</span>
                    </li>
                    <li className="flex items-start">
                      <span className="bg-blue-100 text-blue-800 p-2 rounded mr-3">Tailwind CSS</span>
                      <span className="text-gray-700">Utility-first CSS framework</span>
                    </li>
                  </ul>
                </div>

                <div className="bg-gray-50 p-6 rounded-lg">
                  <h3 className="text-xl font-semibold mb-3 text-blue-600">Backend</h3>
                  <ul className="space-y-3">
                    <li className="flex items-start">
                      <span className="bg-blue-100 text-blue-800 p-2 rounded mr-3">FastAPI</span>
                      <span className="text-gray-700">Modern, fast web framework for building APIs with Python</span>
                    </li>
                    <li className="flex items-start">
                      <span className="bg-blue-100 text-blue-800 p-2 rounded mr-3">MongoDB</span>
                      <span className="text-gray-700">NoSQL database for storing user data and analysis results</span>
                    </li>
                  </ul>
                </div>

                <div className="bg-gray-50 p-6 rounded-lg">
                  <h3 className="text-xl font-semibold mb-3 text-blue-600">Machine Learning</h3>
                  <ul className="space-y-3">
                    <li className="flex items-start">
                      <span className="bg-blue-100 text-blue-800 p-2 rounded mr-3">Scikit-learn</span>
                      <span className="text-gray-700">Machine learning library for Python</span>
                    </li>
                    <li className="flex items-start">
                      <span className="bg-blue-100 text-blue-800 p-2 rounded mr-3">Pandas</span>
                      <span className="text-gray-700">Data manipulation and analysis library</span>
                    </li>
                    <li className="flex items-start">
                      <span className="bg-blue-100 text-blue-800 p-2 rounded mr-3">NumPy</span>
                      <span className="text-gray-700">Numerical computing library</span>
                    </li>
                  </ul>
                </div>

                <div className="bg-gray-50 p-6 rounded-lg">
                  <h3 className="text-xl font-semibold mb-3 text-blue-600">Visualization</h3>
                  <ul className="space-y-3">
                    <li className="flex items-start">
                      <span className="bg-blue-100 text-blue-800 p-2 rounded mr-3">Matplotlib</span>
                      <span className="text-gray-700">Comprehensive library for creating static, animated, and interactive visualizations</span>
                    </li>
                    <li className="flex items-start">
                      <span className="bg-blue-100 text-blue-800 p-2 rounded mr-3">Seaborn</span>
                      <span className="text-gray-700">Statistical data visualization library</span>
                    </li>
                  </ul>
                </div>

                <div className="bg-gray-50 p-6 rounded-lg col-span-1 md:col-span-2">
                  <h3 className="text-xl font-semibold mb-3 text-blue-600">LLM Integration</h3>
                  <ul className="space-y-3">
                    <li className="flex items-start">
                      <span className="bg-blue-100 text-blue-800 p-2 rounded mr-3">Google's Gemini API</span>
                      <span className="text-gray-700">Large Language Model for descriptive analysis and natural language insights</span>
                    </li>
                  </ul>
                </div>
              </div>
            </section>
          )}

          {/* User Guide Section */}
          {activeSection === 'userguide' && (
            <section>
              <h2 className="text-2xl font-bold mb-6 text-blue-700">User Guide</h2>
              <div className="space-y-8">
                <div className="bg-gray-50 p-6 rounded-lg">
                  <h3 className="text-xl font-semibold mb-3 text-blue-600">Getting Started</h3>
                  <ol className="list-decimal pl-6 space-y-4 text-gray-700">
                    <li>
                      <span className="font-medium">Login:</span> Access the platform using your credentials
                    </li>
                    <li>
                      <span className="font-medium">Upload Data:</span> 
                      <ul className="list-disc pl-6 mt-2 space-y-1">
                        <li>Click on "Upload CSV" button or drag and drop your file</li>
                        <li>Supported format: CSV files with structured data</li>
                        <li>File size limit: 100MB</li>
                      </ul>
                    </li>
                    <li>
                      <span className="font-medium">Data Preview:</span> After upload, view the first few rows of your dataset to verify correct loading
                    </li>
                  </ol>
                </div>

                <div className="bg-gray-50 p-6 rounded-lg">
                  <h3 className="text-xl font-semibold mb-3 text-blue-600">LLM Analysis</h3>
                  <ol className="list-decimal pl-6 space-y-4 text-gray-700">
                    <li>Navigate to the LLM Analysis tab (default view after login)</li>
                    <li>Enter your natural language query about the dataset in the input field</li>
                    <li>Click "Analyze" to generate insights</li>
                    <li>View the generated summary and any accompanying visualizations</li>
                    <li>Example queries:
                      <ul className="list-disc pl-6 mt-2 space-y-1">
                        <li>"Show the distribution of ages in this dataset"</li>
                        <li>"What is the correlation between income and purchase frequency?"</li>
                        <li>"Identify any outliers in the sales data"</li>
                      </ul>
                    </li>
                  </ol>
                </div>

                <div className="bg-gray-50 p-6 rounded-lg">
                  <h3 className="text-xl font-semibold mb-3 text-blue-600">Custom Analysis</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-medium mb-2 text-blue-500">Predictive Analysis</h4>
                      <ol className="list-decimal pl-6 space-y-2 text-gray-700 text-sm">
                        <li>Select "Predictive Analysis" from the Custom Analysis tab</li>
                        <li>Choose your target variable (what you want to predict)</li>
                        <li>Select relevant features (variables to use for prediction)</li>
                        <li>Choose a machine learning model:
                          <ul className="list-disc pl-6 mt-1">
                            <li>Linear Regression for continuous values</li>
                            <li>Logistic Regression for binary classification</li>
                            <li>Random Forest for complex relationships</li>
                          </ul>
                        </li>
                        <li>Click "Run Analysis" and view results</li>
                      </ol>
                    </div>
                    <div>
                      <h4 className="font-medium mb-2 text-blue-500">Diagnostic Analysis</h4>
                      <ol className="list-decimal pl-6 space-y-2 text-gray-700 text-sm">
                        <li>Select "Diagnostic Analysis" from the Custom Analysis tab</li>
                        <li>Optionally select a target variable for root cause analysis</li>
                        <li>Choose analysis type:
                          <ul className="list-disc pl-6 mt-1">
                            <li>Correlation matrix</li>
                            <li>Anomaly detection</li>
                            <li>Root cause analysis</li>
                          </ul>
                        </li>
                        <li>Click "Run Analysis" and view results</li>
                      </ol>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 p-6 rounded-lg">
                  <h3 className="text-xl font-semibold mb-3 text-blue-600">Custom Visualization</h3>
                  <ol className="list-decimal pl-6 space-y-4 text-gray-700">
                    <li>Navigate to the Custom Visualization tab</li>
                    <li>Select your desired chart type from the dropdown</li>
                    <li>Choose variables for X and Y axes (where applicable)</li>
                    <li>Customize colors, labels, and other visual elements</li>
                    <li>Click "Generate Visualization" to create your chart</li>
                    <li>Interact with the visualization (hover for details, zoom, etc.)</li>
                    <li>Export the visualization as an image if needed</li>
                  </ol>
                </div>
              </div>
            </section>
          )}

          {/* Limitations Section */}
          {activeSection === 'limitations' && (
            <section>
              <h2 className="text-2xl font-bold mb-6 text-blue-700">Current Limitations</h2>
              <div className="space-y-6">
                <div className="bg-red-50 p-6 rounded-lg border border-red-200">
                  <h3 className="text-xl font-semibold mb-3 text-red-600">File Format Support</h3>
                  <p className="text-gray-700">
                    Currently, Analyza primarily supports datasets in CSV (Comma-Separated Values) format. 
                    Users who work with Excel files (.xlsx), JSON, or database connections may face difficulties.
                  </p>
                  <div className="mt-3 p-3 bg-white rounded border">
                    <h4 className="font-medium text-red-500">Workaround:</h4>
                    <p className="text-gray-700 text-sm">
                      Convert your data to CSV format before uploading. Most spreadsheet and database tools 
                      provide export options to CSV.
                    </p>
                  </div>
                </div>

                <div className="bg-red-50 p-6 rounded-lg border border-red-200">
                  <h3 className="text-xl font-semibold mb-3 text-red-600">API Dependencies</h3>
                  <p className="text-gray-700">
                    Analyza leverages Google's Gemini API for LLM-based descriptive analysis. 
                    This introduces several potential challenges:
                  </p>
                  <ul className="list-disc pl-6 mt-2 space-y-1 text-gray-700">
                    <li>Service availability depends on the external provider</li>
                    <li>May incur costs with high usage volumes</li>
                    <li>Data privacy concerns when sending data to external APIs</li>
                  </ul>
                </div>

                <div className="bg-red-50 p-6 rounded-lg border border-red-200">
                  <h3 className="text-xl font-semibold mb-3 text-red-600">Performance with Large Datasets</h3>
                  <p className="text-gray-700">
                    The current implementation may face performance bottlenecks with very large datasets:
                  </p>
                  <ul className="list-disc pl-6 mt-2 space-y-1 text-gray-700">
                    <li>Memory consumption can be high with complex analyses</li>
                    <li>Processing times may increase significantly</li>
                    <li>Visualizations may become less responsive</li>
                  </ul>
                  <div className="mt-3 p-3 bg-white rounded border">
                    <h4 className="font-medium text-red-500">Recommendation:</h4>
                    <p className="text-gray-700 text-sm">
                      For large datasets, consider sampling your data or breaking it into smaller chunks 
                      for analysis. We're working on optimizations for better large dataset handling.
                    </p>
                  </div>
                </div>

                <div className="bg-blue-50 p-6 rounded-lg border border-blue-200">
                  <h3 className="text-xl font-semibold mb-3 text-blue-600">Future Enhancements</h3>
                  <p className="text-gray-700 mb-4">
                    We're actively working to address these limitations in future releases:
                  </p>
                  <ul className="list-disc pl-6 space-y-2 text-gray-700">
                    <li>Support for additional file formats (Excel, JSON, database connections)</li>
                    <li>On-premise LLM options to reduce API dependencies</li>
                    <li>Distributed computing support for large dataset processing</li>
                    <li>Advanced caching mechanisms for improved performance</li>
                    <li>Real-time data processing capabilities</li>
                  </ul>
                </div>
              </div>
            </section>
          )}
        </main>
      </div>

      {/* Footer */}
      <footer className="bg-gray-800 text-white p-6 mt-12">
        <div className="container mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-4 md:mb-0">
              <h3 className="text-xl font-bold">Analyza</h3>
              <p className="text-gray-400">Data Analysis & Visualization Tool</p>
            </div>
            <div className="text-sm text-gray-400">
              <p>Developed by  Shruti Srivastava, Dhuruv Kumar, Khushi Chauhan</p>
              <p>Under the guidance of Dr. Deepak Kumar Sharma</p>
              <p>School of Computer Science, UPES - March 2025</p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Documentation;