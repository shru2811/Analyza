import React from "react";

const LandingPage = ({ setActiveTab }) => {
  return (
    <div>
      {/* Hero Section */}
      <div className="py-12 bg-gradient-to-r from-blue-700 from-10% via-blue-400 via-40% to-emerald-500 to-100%">
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
                onClick={() => setActiveTab("LLM")}
                className="px-8 py-3 border border-transparent text-base font-medium rounded-md text-indigo-700 bg-white hover:bg-gray-50 md:py-4 md:text-lg md:px-10"
              >
                Get Started
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Feature Cards */}
      <div className="py-12 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:text-center">
            <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
              Comprehensive Data Analysis Tools
            </p>
            <p className="mt-4 max-w-2xl text-xl text-gray-500 lg:mx-auto">
              Transform your raw data into actionable insights with our suite of analysis modules
            </p>
          </div>

          <div className="mt-10">
            <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-3">
              {/* LLM Analysis */}
              <div className="pt-6 cursor-pointer" onClick={() => setActiveTab("LLM")}>
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

              {/* Predictive & Diagnostic */}
              <div className="pt-6 cursor-pointer" onClick={() => setActiveTab("CustomAnalysis")}>
                <div className="h-full rounded-lg border-2 border-gray-200 border-opacity-60 overflow-hidden hover:border-indigo-500 transition-colors duration-300">
                  <div className="p-6">
                    <div className="w-10 h-10 inline-flex items-center justify-center rounded-full bg-indigo-100 text-indigo-500 mb-4">
                      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                      </svg>
                    </div>
                    <h2 className="text-lg text-gray-900 font-medium title-font mb-2">Predictive & Diagnostic Analysis</h2>
                    <p className="leading-relaxed text-base">
                      Forecast future trends and diagnose root causes with our custom analytical modules
                    </p>
                  </div>
                </div>
              </div>

              {/* Custom Visualization */}
              <div className="pt-6 cursor-pointer" onClick={() => setActiveTab("CustomVisualization")}>
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

      {/* Benefits Section */}
      <div className="py-12 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:text-center mb-10">
            <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
              Why Choose Analyza
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <div className="flex items-center mb-4">
                <div className="bg-indigo-100 rounded-full p-2 mr-4">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <h3 className="text-xl font-medium text-gray-900">Descriptive Analysis (LLM-based)</h3>
              </div>
              <p className="text-gray-600">Integrated Gemini LLM API for analysis which gives supportive summary and visualization on the user query and data given by user.</p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <div className="flex items-center mb-4">
                <div className="bg-indigo-100 rounded-full p-2 mr-4">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                  </svg>
                </div>
                <h3 className="text-xl font-medium text-gray-900">Predictive Analysis</h3>
              </div>
              <p className="text-gray-600">Made using custom modules (using Python libraries) on inputting the data set allows for suggestions to choose target, features and model to be selected for prediction.</p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <div className="flex items-center mb-4">
                <div className="bg-indigo-100 rounded-full p-2 mr-4">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
                  </svg>
                </div>
                <h3 className="text-xl font-medium text-gray-900">Diagnostic Analysis</h3>
              </div>
              <p className="text-gray-600">Made using custom modules (using Python libraries) on inputting the data set allows root cause analysis and anomaly detection.</p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <div className="flex items-center mb-4">
                <div className="bg-indigo-100 rounded-full p-2 mr-4">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <h3 className="text-xl font-medium text-gray-900">Custom Visualizations</h3>
              </div>
              <p className="text-gray-600">Input data set by user and allows 7 types of choices (Bar Chart, Scatter Plot, Line Graph, Histogram, Box Plot, Violin Plot, Pie Chart) to make visualization of and have user option to choose the x and y axis.</p>
            </div>
          </div>
        </div>
      </div>

      {/* Call to Action */}
      <div className="bg-indigo-700">
        <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8 lg:py-16">
          <div className="lg:grid lg:grid-cols-2 lg:gap-8 lg:items-center">
            <div>
              <h2 className="text-3xl font-extrabold text-white sm:text-4xl">
                Ready to transform your data?
              </h2>
              <p className="mt-3 max-w-3xl text-lg text-indigo-200">
                Start uncovering insights from your data in minutes. No complex setups, no waiting.
              </p>
              <div className="mt-8">
                <div className="inline-flex rounded-md shadow">
                  <button
                    onClick={() => setActiveTab("LLM")}
                    className="inline-flex items-center justify-center px-5 py-3 border border-transparent text-base font-medium rounded-md text-indigo-700 bg-white hover:bg-indigo-50"
                  >
                    Get Started Now
                  </button>
                </div>
              </div>
            </div>
            <div className="mt-8 grid grid-cols-2 gap-0.5 md:grid-cols-3 lg:mt-0 lg:grid-cols-2">
              <div className="col-span-1 flex justify-center py-8 px-8 bg-indigo-600">
                <p className="text-center text-white text-3xl font-bold">Fast</p>
              </div>
              <div className="col-span-1 flex justify-center py-8 px-8 bg-indigo-600">
                <p className="text-center text-white text-3xl font-bold">Smart Insights</p>
              </div>
              <div className="col-span-1 flex justify-center py-8 px-8 bg-indigo-600">
                <p className="text-center text-white text-3xl font-bold">Infographics</p>
              </div>
              <div className="col-span-1 flex justify-center py-8 px-8 bg-indigo-600">
                <p className="text-center text-white text-3xl font-bold">Accurate</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
