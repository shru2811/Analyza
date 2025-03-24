import React, { useState } from 'react';

const App = () => {
  const [activeTab, setActiveTab] = useState('home');

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation Bar */}
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex-shrink-0 flex items-center">
              <span className="text-2xl font-bold text-indigo-600">Analyza</span>
            </div>
            <div className="flex items-center space-x-4">
              <button 
                onClick={() => setActiveTab('predictive')}
                className={`px-3 py-2 rounded-md text-sm font-medium ${
                  activeTab === 'predictive' ? 'bg-indigo-100 text-indigo-700' : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                Predictive & Diagnostic
              </button>
              <button 
                onClick={() => setActiveTab('descriptive')}
                className={`px-3 py-2 rounded-md text-sm font-medium ${
                  activeTab === 'descriptive' ? 'bg-indigo-100 text-indigo-700' : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                Descriptive Analysis
              </button>
              <button 
                onClick={() => setActiveTab('visualization')}
                className={`px-3 py-2 rounded-md text-sm font-medium ${
                  activeTab === 'visualization' ? 'bg-indigo-100 text-indigo-700' : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                Custom Visualization
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      {activeTab === 'home' && (
        <div className="py-12 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center">
              <h1 className="text-4xl font-extrabold text-white sm:text-5xl sm:tracking-tight lg:text-6xl">
                Unlock the Power of Your Data
              </h1>
              <p className="mt-5 max-w-xl mx-auto text-xl text-white">
                Advanced analytics platform with predictive, diagnostic, and descriptive analysis capabilities
              </p>
              <div className="mt-8 flex justify-center">
                <button
                  onClick={() => setActiveTab('predictive')}
                  className="px-8 py-3 border border-transparent text-base font-medium rounded-md text-indigo-700 bg-white hover:bg-gray-50 md:py-4 md:text-lg md:px-10"
                >
                  Get Started
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Feature Cards */}
      {activeTab === 'home' && (
        <div className="py-12 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="lg:text-center">
              <h2 className="text-base text-indigo-600 font-semibold tracking-wide uppercase">Features</h2>
              <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
                Comprehensive Data Analysis Tools
              </p>
              <p className="mt-4 max-w-2xl text-xl text-gray-500 lg:mx-auto">
                Transform your raw data into actionable insights with our suite of analysis modules
              </p>
            </div>

            <div className="mt-10">
              <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-3">
                {/* Predictive & Diagnostic */}
                <div className="pt-6 cursor-pointer" onClick={() => setActiveTab('predictive')}>
                  <div className="h-full rounded-lg border-2 border-gray-200 border-opacity-60 overflow-hidden hover:border-indigo-500 transition-colors duration-300">
                    <div className="p-6">
                      <div className="w-10 h-10 inline-flex items-center justify-center rounded-full bg-indigo-100 text-indigo-500 mb-4">
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                        </svg>
                      </div>
                      <h2 className="text-lg text-gray-900 font-medium title-font mb-2">Predictive & Diagnostic</h2>
                      <p className="leading-relaxed text-base">
                        Forecast future trends and diagnose root causes with our custom analytical modules
                      </p>
                    </div>
                  </div>
                </div>

                {/* Descriptive Analysis */}
                <div className="pt-6 cursor-pointer" onClick={() => setActiveTab('descriptive')}>
                  <div className="h-full rounded-lg border-2 border-gray-200 border-opacity-60 overflow-hidden hover:border-indigo-500 transition-colors duration-300">
                    <div className="p-6">
                      <div className="w-10 h-10 inline-flex items-center justify-center rounded-full bg-indigo-100 text-indigo-500 mb-4">
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"></path>
                        </svg>
                      </div>
                      <h2 className="text-lg text-gray-900 font-medium title-font mb-2">Descriptive Analysis</h2>
                      <p className="leading-relaxed text-base">
                        LLM-powered analysis that provides detailed insights and explanations of your data
                      </p>
                    </div>
                  </div>
                </div>

                {/* Custom Visualization */}
                <div className="pt-6 cursor-pointer" onClick={() => setActiveTab('visualization')}>
                  <div className="h-full rounded-lg border-2 border-gray-200 border-opacity-60 overflow-hidden hover:border-indigo-500 transition-colors duration-300">
                    <div className="p-6">
                      <div className="w-10 h-10 inline-flex items-center justify-center rounded-full bg-indigo-100 text-indigo-500 mb-4">
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z"></path>
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z"></path>
                        </svg>
                      </div>
                      <h2 className="text-lg text-gray-900 font-medium title-font mb-2">Custom Visualization</h2>
                      <p className="leading-relaxed text-base">
                        Create beautiful, interactive visualizations from your uploaded datasets
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Predictive & Diagnostic Module */}
      {activeTab === 'predictive' && (
        <div className="py-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-12">
              <h1 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
                Predictive & Diagnostic Analysis
              </h1>
              <p className="mt-4 text-lg text-gray-500">
                Forecast trends and identify root causes with our advanced algorithms
              </p>
            </div>

            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <div className="mb-6">
                  <label htmlFor="dataset" className="block text-sm font-medium text-gray-700">Upload your dataset</label>
                  <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
                    <div className="space-y-1 text-center">
                      <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                        <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                      <div className="flex text-sm text-gray-600">
                        <label htmlFor="file-upload" className="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500">
                          <span>Upload a file</span>
                          <input id="file-upload" name="file-upload" type="file" className="sr-only" />
                        </label>
                        <p className="pl-1">or drag and drop</p>
                      </div>
                      <p className="text-xs text-gray-500">CSV, XLSX up to 10MB</p>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                  <div>
                    <label htmlFor="analysis-type" className="block text-sm font-medium text-gray-700">Analysis Type</label>
                    <select id="analysis-type" name="analysis-type" className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md">
                      <option>Time Series Forecasting</option>
                      <option>Regression Analysis</option>
                      <option>Classification</option>
                      <option>Anomaly Detection</option>
                    </select>
                  </div>

                  <div>
                    <label htmlFor="target-variable" className="block text-sm font-medium text-gray-700">Target Variable</label>
                    <input type="text" name="target-variable" id="target-variable" className="mt-1 focus:ring-indigo-500 focus:border-indigo-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md" placeholder="e.g. revenue, conversion_rate" />
                  </div>
                </div>

                <div className="mt-6">
                  <button type="submit" className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                    Run Analysis
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Descriptive Analysis Module */}
      {activeTab === 'descriptive' && (
        <div className="py-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-12">
              <h1 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
                LLM-Powered Descriptive Analysis
              </h1>
              <p className="mt-4 text-lg text-gray-500">
                Get natural language insights and explanations from your data
              </p>
            </div>

            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <div className="mb-6">
                  <label htmlFor="dataset" className="block text-sm font-medium text-gray-700">Upload your dataset</label>
                  <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
                    <div className="space-y-1 text-center">
                      <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                        <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                      <div className="flex text-sm text-gray-600">
                        <label htmlFor="file-upload-llm" className="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500">
                          <span>Upload a file</span>
                          <input id="file-upload-llm" name="file-upload-llm" type="file" className="sr-only" />
                        </label>
                        <p className="pl-1">or drag and drop</p>
                      </div>
                      <p className="text-xs text-gray-500">CSV, XLSX up to 10MB</p>
                    </div>
                  </div>
                </div>

                <div className="mb-6">
                  <label htmlFor="prompt" className="block text-sm font-medium text-gray-700">What would you like to know about your data?</label>
                  <div className="mt-1">
                    <textarea id="prompt" name="prompt" rows="3" className="shadow-sm focus:ring-indigo-500 focus:border-indigo-500 block w-full sm:text-sm border border-gray-300 rounded-md" placeholder="e.g. Summarize the main trends in my sales data"></textarea>
                  </div>
                </div>

                <div>
                  <button type="submit" className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                    Generate Insights
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Custom Visualization Module */}
      {activeTab === 'visualization' && (
        <div className="py-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-12">
              <h1 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
                Custom Data Visualization
              </h1>
              <p className="mt-4 text-lg text-gray-500">
                Create beautiful, interactive charts and visualizations
              </p>
            </div>

            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <div className="mb-6">
                  <label htmlFor="dataset" className="block text-sm font-medium text-gray-700">Upload your dataset</label>
                  <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
                    <div className="space-y-1 text-center">
                      <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                        <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                      <div className="flex text-sm text-gray-600">
                        <label htmlFor="file-upload-viz" className="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500">
                          <span>Upload a file</span>
                          <input id="file-upload-viz" name="file-upload-viz" type="file" className="sr-only" />
                        </label>
                        <p className="pl-1">or drag and drop</p>
                      </div>
                      <p className="text-xs text-gray-500">CSV, XLSX up to 10MB</p>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                  <div>
                    <label htmlFor="chart-type" className="block text-sm font-medium text-gray-700">Chart Type</label>
                    <select id="chart-type" name="chart-type" className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md">
                      <option>Line Chart</option>
                      <option>Bar Chart</option>
                      <option>Scatter Plot</option>
                      <option>Pie Chart</option>
                      <option>Heat Map</option>
                      <option>Box Plot</option>
                    </select>
                  </div>

                  <div>
                    <label htmlFor="data-fields" className="block text-sm font-medium text-gray-700">Select Data Fields</label>
                    <input type="text" name="data-fields" id="data-fields" className="mt-1 focus:ring-indigo-500 focus:border-indigo-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md" placeholder="e.g. date, sales, category" />
                  </div>
                </div>

                <div className="mt-6">
                  <button type="submit" className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                    Generate Visualization
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="bg-white">
        <div className="max-w-7xl mx-auto py-12 px-4 overflow-hidden sm:px-6 lg:px-8">
          <p className="mt-8 text-center text-base text-gray-400">
            &copy; 2025 Analyza. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;