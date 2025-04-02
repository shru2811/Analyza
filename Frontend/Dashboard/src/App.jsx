import React, { useState } from "react";
import { AnalyzaLogo } from "./AnalyzaLogo";
import LandingPage from "./LandingPage";
import AnalysisInterface from "./AnalysisInterface";
import { UPESLogo } from "./UPESLogo";

const App = () => {
  const [activeTab, setActiveTab] = useState("home");

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-lg">
        <div className="max-w-8xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <AnalyzaLogo />
            <div className="space-x-1">
              <button
                className={`px-4 py-2 text-lg font-semibold ${activeTab === "home" ? "text-indigo-600 border-b-2 border-indigo-600" : "text-gray-600"}`}
                onClick={() => setActiveTab("home")}
              >
                Home
              </button>
              <button
                className={`px-4 py-2 text-lg font-semibold ${activeTab === "LLM" ? "text-indigo-600 border-b-2 border-indigo-600" : "text-gray-600"}`}
                onClick={() => setActiveTab("LLM")}
              >
                LLM Analysis
              </button>
              <button
                className={`px-4 py-2 text-lg font-semibold ${activeTab === "CustomAnalysis" ? "text-indigo-600 border-b-2 border-indigo-600" : "text-gray-600"}`}
                onClick={() => setActiveTab("CustomAnalysis")}
              >
                Custom Analysis
              </button>
              <button
                className={`px-4 py-2 text-lg font-semibold ${activeTab === "CustomVisualization" ? "text-indigo-600 border-b-2 border-indigo-600" : "text-gray-600"}`}
                onClick={() => setActiveTab("CustomVisualization")}
              >
                Custom Visualization
              </button>
              <button
                className={`px-4 py-2 text-lg font-semibold ${activeTab === "Documentation" ? "text-indigo-600 border-b-2 border-indigo-600" : "text-gray-600"}`}
                onClick={() => setActiveTab("Documentation")}
              >
                Documentation
              </button>  
              <button
                className={`px-4 py-2 text-lg font-semibold ${activeTab === "About" ? "text-indigo-600 border-b-2 border-indigo-600" : "text-gray-600"}`}
                onClick={() => setActiveTab("About")}
              >
                About
              </button>  
            </div>
            <UPESLogo/>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      {activeTab === "home" ? (
        <LandingPage setActiveTab={setActiveTab} />
      ) : (
        <AnalysisInterface analysisType={activeTab} />
      )}

      {/* Footer */}
      <footer className="bg-white mt-12">
        <div className="max-w-7xl mx-auto py-8 px-4 overflow-hidden sm:px-6 lg:px-8">
          <p className="text-center text-base text-gray-400">
            &copy; {new Date().getFullYear()} Analyza. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;