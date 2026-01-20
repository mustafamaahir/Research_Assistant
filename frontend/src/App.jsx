import React, { useState, useEffect } from 'react';
import { Upload, FileText, BarChart3, Trash2, Download, CheckCircle, XCircle, PlusCircle, Loader, AlertCircle } from 'lucide-react';

const ResearchAssistant = () => {
  const [activeTab, setActiveTab] = useState('research');
  const [sessionId, setSessionId] = useState(null);
  const [researchFiles, setResearchFiles] = useState([]);
  const [analysisFiles, setAnalysisFiles] = useState({
    data: null,
    guide: null,
    objectives: null
  });
  const [researchPrompt, setResearchPrompt] = useState('');
  const [researchResponse, setResearchResponse] = useState('');
  const [analysisStatus, setAnalysisStatus] = useState('idle');
  const [analysisSteps, setAnalysisSteps] = useState([]);
  const [currentStep, setCurrentStep] = useState(null);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState('');
  const [error, setError] = useState('');
  const [reportSections, setReportSections] = useState({
    research: ['Introduction', 'Literature Review', 'Methodology', 'Results', 'Discussion', 'Conclusion'],
    analysis: ['Objectives', 'Methods', 'Data Description', 'Statistical Analysis', 'Results', 'Tables & Figures', 'Interpretation', 'Conclusion']
  });
  const [selectedSections, setSelectedSections] = useState([]);
  const [customSection, setCustomSection] = useState('');

  // Update API_BASE with your deployed Render URL
  // For local testing: 'http://localhost:8000'
  // For production: 'https://your-app.onrender.com'
  const API_BASE = 'http://localhost:8000';

  useEffect(() => {
    if (!sessionId) {
      setSessionId(generateSessionId());
    }
  }, [sessionId]);

  const generateSessionId = () => {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  };

  const handleResearchFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    setLoading(true);
    setUploadProgress('Uploading PDFs...');
    setError('');
    
    try {
      const response = await fetch(`${API_BASE}/research/upload`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) throw new Error('Upload failed');
      
      const data = await response.json();
      setSessionId(data.session_id);
      setResearchFiles(prev => [...prev, ...data.files]);
      setUploadProgress('');
    } catch (error) {
      console.error('Upload error:', error);
      setError('Failed to upload PDFs. Please try again.');
    }
    setLoading(false);
  };

  const handleAnalysisFileUpload = async (type, e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!sessionId) {
      setSessionId(generateSessionId());
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', type);
    formData.append('session_id', sessionId);

    setLoading(true);
    setUploadProgress(`Uploading ${type} file...`);
    setError('');
    
    try {
      const response = await fetch(`${API_BASE}/analysis/upload`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) throw new Error('Upload failed');
      
      const data = await response.json();
      setAnalysisFiles(prev => ({ ...prev, [type]: data.file }));
      setUploadProgress('');
    } catch (error) {
      console.error('Upload error:', error);
      setError(`Failed to upload ${type} file. Please try again.`);
    }
    setLoading(false);
  };

  const handleResearchTask = async () => {
    if (!researchPrompt.trim() || !sessionId) return;

    setLoading(true);
    setError('');
    
    try {
      const response = await fetch(`${API_BASE}/research/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          prompt: researchPrompt,
          session_id: sessionId 
        })
      });
      
      if (!response.ok) throw new Error('Task execution failed');
      
      const data = await response.json();
      setResearchResponse(data.response);
    } catch (error) {
      console.error('Task error:', error);
      setError('Failed to execute task. Make sure PDFs are uploaded and processed.');
    }
    setLoading(false);
  };

  const startAnalysis = async () => {
    if (!analysisFiles.data || !analysisFiles.objectives) {
      setError('Please upload data file and objectives file');
      return;
    }

    setLoading(true);
    setAnalysisStatus('planning');
    setError('');
    
    try {
      const formData = new FormData();
      formData.append('session_id', sessionId);
      
      const response = await fetch(`${API_BASE}/analysis/start`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) throw new Error('Analysis start failed');
      
      const data = await response.json();
      setAnalysisSteps(data.steps);
      setCurrentStep(data.steps[0]);
      setCurrentStepIndex(0);
      setAnalysisStatus('awaiting_approval');
    } catch (error) {
      console.error('Analysis error:', error);
      setError('Failed to start analysis. Please check your files.');
      setAnalysisStatus('idle');
    }
    setLoading(false);
  };

  const approveStep = async (approved) => {
    if (!currentStep) return;

    setLoading(true);
    setError('');
    
    try {
      const response = await fetch(`${API_BASE}/analysis/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          step_id: currentStep.id, 
          approved,
          session_id: sessionId
        })
      });
      
      if (!response.ok) throw new Error('Approval failed');
      
      const data = await response.json();
      
      if (approved && data.result) {
        const updatedSteps = [...analysisSteps];
        updatedSteps[currentStepIndex] = { 
          ...currentStep, 
          completed: true, 
          result: data.result 
        };
        setAnalysisSteps(updatedSteps);
        
        const nextIndex = currentStepIndex + 1;
        if (nextIndex < analysisSteps.length) {
          setCurrentStep(analysisSteps[nextIndex]);
          setCurrentStepIndex(nextIndex);
        } else {
          setAnalysisStatus('completed');
          setCurrentStep(null);
        }
      } else {
        setAnalysisStatus('rejected');
        setError('Analysis step rejected. You can restart the analysis.');
      }
    } catch (error) {
      console.error('Approval error:', error);
      setError('Failed to process approval. Please try again.');
    }
    setLoading(false);
  };

  const generateReport = async (type) => {
    const sections = selectedSections.length > 0 ? selectedSections : reportSections[type];
    
    setLoading(true);
    setError('');
    
    try {
      const response = await fetch(`${API_BASE}/report/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          type, 
          sections,
          session_id: sessionId
        })
      });
      
      if (!response.ok) throw new Error('Report generation failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${type}_report_${new Date().toISOString().split('T')[0]}.docx`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Report generation error:', error);
      setError('Failed to generate report. Please try again.');
    }
    setLoading(false);
  };

  const clearSession = async () => {
    if (!confirm('Clear all session data? This cannot be undone.')) return;
    
    setLoading(true);
    
    try {
      if (sessionId) {
        const formData = new FormData();
        formData.append('session_id', sessionId);
        
        await fetch(`${API_BASE}/session/clear`, { 
          method: 'POST',
          body: formData
        });
      }
      
      setSessionId(generateSessionId());
      setResearchFiles([]);
      setAnalysisFiles({ data: null, guide: null, objectives: null });
      setResearchPrompt('');
      setResearchResponse('');
      setAnalysisSteps([]);
      setCurrentStep(null);
      setCurrentStepIndex(0);
      setAnalysisStatus('idle');
      setSelectedSections([]);
      setError('');
      setUploadProgress('');
    } catch (error) {
      console.error('Clear error:', error);
      setError('Failed to clear session.');
    }
    setLoading(false);
  };

  const toggleSection = (section) => {
    setSelectedSections(prev => 
      prev.includes(section) ? prev.filter(s => s !== section) : [...prev, section]
    );
  };

  const addCustomSection = () => {
    if (customSection.trim()) {
      setSelectedSections(prev => [...prev, customSection.trim()]);
      setCustomSection('');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-teal-600 rounded-lg flex items-center justify-center">
                <FileText className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-800">AI Research Assistant</h1>
                <p className="text-sm text-slate-500">Powered by Groq AI • Professional Research & Analysis</p>
              </div>
            </div>
            <button
              onClick={clearSession}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-colors disabled:opacity-50"
            >
              <Trash2 className="w-4 h-4" />
              Clear Session
            </button>
          </div>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="max-w-7xl mx-auto px-6 mt-4">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-600 mt-0.5" />
            <div className="flex-1">
              <p className="text-red-800 font-medium">Error</p>
              <p className="text-red-600 text-sm">{error}</p>
            </div>
            <button onClick={() => setError('')} className="text-red-400 hover:text-red-600">
              <XCircle className="w-5 h-5" />
            </button>
          </div>
        </div>
      )}

      {/* Upload Progress */}
      {uploadProgress && (
        <div className="max-w-7xl mx-auto px-6 mt-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-center gap-3">
            <Loader className="w-5 h-5 text-blue-600 animate-spin" />
            <p className="text-blue-800">{uploadProgress}</p>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="max-w-7xl mx-auto px-6 mt-6">
        <div className="flex gap-2 border-b border-slate-200">
          <button
            onClick={() => setActiveTab('research')}
            className={`px-6 py-3 font-medium transition-colors relative ${
              activeTab === 'research'
                ? 'text-blue-600'
                : 'text-slate-600 hover:text-slate-800'
            }`}
          >
            <FileText className="w-4 h-4 inline mr-2" />
            Research Assistant
            {activeTab === 'research' && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600"></div>
            )}
          </button>
          <button
            onClick={() => setActiveTab('analysis')}
            className={`px-6 py-3 font-medium transition-colors relative ${
              activeTab === 'analysis'
                ? 'text-teal-600'
                : 'text-slate-600 hover:text-slate-800'
            }`}
          >
            <BarChart3 className="w-4 h-4 inline mr-2" />
            Analysis Agent
            {activeTab === 'analysis' && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-teal-600"></div>
            )}
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 'research' ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left Panel - Upload & Task */}
            <div className="space-y-6">
              <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                <h2 className="text-lg font-semibold text-slate-800 mb-4">Upload Research PDFs</h2>
                <label className="block w-full cursor-pointer">
                  <div className="border-2 border-dashed border-slate-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors">
                    <Upload className="w-12 h-12 mx-auto text-slate-400 mb-3" />
                    <p className="text-slate-600 font-medium">Click to upload PDFs</p>
                    <p className="text-sm text-slate-400 mt-1">Multiple files supported</p>
                  </div>
                  <input
                    type="file"
                    multiple
                    accept=".pdf"
                    onChange={handleResearchFileUpload}
                    className="hidden"
                    disabled={loading}
                  />
                </label>
                
                {researchFiles.length > 0 && (
                  <div className="mt-4 space-y-2">
                    <p className="text-sm text-slate-600 font-medium">{researchFiles.length} file(s) uploaded</p>
                    {researchFiles.map((file, idx) => (
                      <div key={idx} className="flex items-center gap-2 p-2 bg-blue-50 rounded-lg">
                        <FileText className="w-4 h-4 text-blue-600" />
                        <span className="text-sm text-slate-700 flex-1">{file.name}</span>
                        <CheckCircle className="w-4 h-4 text-green-600" />
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                <h2 className="text-lg font-semibold text-slate-800 mb-4">Research Task</h2>
                <textarea
                  value={researchPrompt}
                  onChange={(e) => setResearchPrompt(e.target.value)}
                  placeholder="Enter your research task (e.g., 'Write the introduction based on uploaded papers' or 'Summarize the methodology sections')"
                  className="w-full h-32 px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                  disabled={loading}
                />
                <button
                  onClick={handleResearchTask}
                  disabled={loading || !researchPrompt.trim() || researchFiles.length === 0}
                  className="mt-4 w-full px-6 py-3 bg-gradient-to-r from-blue-600 to-teal-600 text-white rounded-lg font-medium hover:from-blue-700 hover:to-teal-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <Loader className="w-5 h-5 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    'Execute Task'
                  )}
                </button>
              </div>

              <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                <h2 className="text-lg font-semibold text-slate-800 mb-4">Report Sections</h2>
                <div className="space-y-2 mb-4">
                  {reportSections.research.map(section => (
                    <label key={section} className="flex items-center gap-2 cursor-pointer hover:bg-slate-50 p-2 rounded">
                      <input
                        type="checkbox"
                        checked={selectedSections.includes(section)}
                        onChange={() => toggleSection(section)}
                        className="w-4 h-4 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
                      />
                      <span className="text-slate-700">{section}</span>
                    </label>
                  ))}
                </div>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={customSection}
                    onChange={(e) => setCustomSection(e.target.value)}
                    placeholder="Add custom section"
                    className="flex-1 px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    onKeyPress={(e) => e.key === 'Enter' && addCustomSection()}
                  />
                  <button
                    onClick={addCustomSection}
                    disabled={!customSection.trim()}
                    className="px-4 py-2 bg-blue-100 text-blue-600 rounded-lg hover:bg-blue-200 disabled:opacity-50"
                  >
                    <PlusCircle className="w-5 h-5" />
                  </button>
                </div>
                {selectedSections.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {selectedSections.map(section => (
                      <span key={section} className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">
                        {section}
                      </span>
                    ))}
                  </div>
                )}
                <button
                  onClick={() => generateReport('research')}
                  disabled={loading || !sessionId}
                  className="mt-4 w-full px-6 py-3 bg-slate-700 text-white rounded-lg font-medium hover:bg-slate-800 disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Generate Research Report
                </button>
              </div>
            </div>

            {/* Right Panel - Response */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
              <h2 className="text-lg font-semibold text-slate-800 mb-4">AI Response</h2>
              <div className="prose max-w-none">
                {researchResponse ? (
                  <div className="whitespace-pre-wrap text-slate-700 leading-relaxed">{researchResponse}</div>
                ) : (
                  <div className="text-center py-12">
                    <FileText className="w-16 h-16 mx-auto text-slate-300 mb-4" />
                    <p className="text-slate-400 italic">Your research task response will appear here...</p>
                    <p className="text-sm text-slate-400 mt-2">Upload PDFs and enter a task to get started</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {/* File Uploads */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                <h3 className="font-semibold text-slate-800 mb-3 flex items-center gap-2">
                  Data File 
                  <span className="text-xs bg-red-100 text-red-600 px-2 py-0.5 rounded">Required</span>
                </h3>
                <label className="block cursor-pointer">
                  <div className="border-2 border-dashed border-slate-300 rounded-lg p-6 text-center hover:border-teal-400 transition-colors">
                    <Upload className="w-8 h-8 mx-auto text-slate-400 mb-2" />
                    <p className="text-sm text-slate-600">Upload CSV/XLSX</p>
                  </div>
                  <input
                    type="file"
                    accept=".csv,.xlsx"
                    onChange={(e) => handleAnalysisFileUpload('data', e)}
                    className="hidden"
                    disabled={loading}
                  />
                </label>
                {analysisFiles.data && (
                  <p className="mt-2 text-sm text-teal-600 flex items-center gap-1">
                    <CheckCircle className="w-4 h-4" /> {analysisFiles.data.name}
                  </p>
                )}
              </div>

              <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                <h3 className="font-semibold text-slate-800 mb-3 flex items-center gap-2">
                  Format Guide
                  <span className="text-xs bg-slate-100 text-slate-600 px-2 py-0.5 rounded">Optional</span>
                </h3>
                <label className="block cursor-pointer">
                  <div className="border-2 border-dashed border-slate-300 rounded-lg p-6 text-center hover:border-teal-400 transition-colors">
                    <Upload className="w-8 h-8 mx-auto text-slate-400 mb-2" />
                    <p className="text-sm text-slate-600">Upload DOC/DOCX</p>
                  </div>
                  <input
                    type="file"
                    accept=".doc,.docx"
                    onChange={(e) => handleAnalysisFileUpload('guide', e)}
                    className="hidden"
                    disabled={loading}
                  />
                </label>
                {analysisFiles.guide && (
                  <p className="mt-2 text-sm text-teal-600 flex items-center gap-1">
                    <CheckCircle className="w-4 h-4" /> {analysisFiles.guide.name}
                  </p>
                )}
              </div>

              <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                <h3 className="font-semibold text-slate-800 mb-3 flex items-center gap-2">
                  Objectives & Methods
                  <span className="text-xs bg-red-100 text-red-600 px-2 py-0.5 rounded">Required</span>
                </h3>
                <label className="block cursor-pointer">
                  <div className="border-2 border-dashed border-slate-300 rounded-lg p-6 text-center hover:border-teal-400 transition-colors">
                    <Upload className="w-8 h-8 mx-auto text-slate-400 mb-2" />
                    <p className="text-sm text-slate-600">Upload DOC/DOCX</p>
                  </div>
                  <input
                    type="file"
                    accept=".doc,.docx"
                    onChange={(e) => handleAnalysisFileUpload('objectives', e)}
                    className="hidden"
                    disabled={loading}
                  />
                </label>
                {analysisFiles.objectives && (
                  <p className="mt-2 text-sm text-teal-600 flex items-center gap-1">
                    <CheckCircle className="w-4 h-4" /> {analysisFiles.objectives.name}
                  </p>
                )}
              </div>
            </div>

            {/* Start Analysis Button */}
            {analysisStatus === 'idle' && (
              <div className="text-center">
                <button
                  onClick={startAnalysis}
                  disabled={loading || !analysisFiles.data || !analysisFiles.objectives}
                  className="px-8 py-4 bg-gradient-to-r from-teal-600 to-blue-600 text-white rounded-lg font-medium text-lg hover:from-teal-700 hover:to-blue-700 disabled:opacity-50 transition-all shadow-lg flex items-center gap-3 mx-auto"
                >
                  {loading ? (
                    <>
                      <Loader className="w-6 h-6 animate-spin" />
                      Initializing Analysis...
                    </>
                  ) : (
                    <>
                      <BarChart3 className="w-6 h-6" />
                      Start AI Analysis
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Analysis Steps */}
            {analysisSteps.length > 0 && (
              <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-lg font-semibold text-slate-800">Analysis Progress</h2>
                  <span className="text-sm text-slate-500">
                    Step {currentStepIndex + 1} of {analysisSteps.length}
                  </span>
                </div>
                
                <div className="space-y-4">
                  {analysisSteps.map((step, idx) => (
                    <div
                      key={step.id}
                      className={`p-4 rounded-lg border-2 transition-all ${
                        step.completed
                          ? 'bg-green-50 border-green-200'
                          : currentStep?.id === step.id
                          ? 'bg-blue-50 border-blue-300 shadow-md'
                          : 'bg-slate-50 border-slate-200 opacity-60'
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-xs font-bold text-slate-500">STEP {idx + 1}</span>
                            {step.completed && <CheckCircle className="w-5 h-5 text-green-600" />}
                          </div>
                          <h3 className="font-medium text-slate-800 mb-1">{step.title}</h3>
                          <p className="text-sm text-slate-600">{step.description}</p>
                          
                          {step.result && (
                            <div className="mt-3 p-3 bg-white rounded border border-slate-200">
                              <p className="text-xs font-semibold text-slate-500 mb-1">RESULT:</p>
                              <p className="text-sm text-slate-700 whitespace-pre-wrap">{step.result}</p>
                            </div>
                          )}
                        </div>
                      </div>
                      
                      {currentStep?.id === step.id && analysisStatus === 'awaiting_approval' && (
                        <div className="mt-4 flex gap-3">
                          <button
                            onClick={() => approveStep(true)}
                            disabled={loading}
                            className="flex-1 px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 flex items-center justify-center gap-2 font-medium transition-colors"
                          >
                            {loading ? (
                              <Loader className="w-4 h-4 animate-spin" />
                            ) : (
                              <>
                                <CheckCircle className="w-5 h-5" />
                                Approve & Continue
                              </>
                            )}
                          </button>
                          <button
                            onClick={() => approveStep(false)}
                            disabled={loading}
                            className="flex-1 px-4 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 flex items-center justify-center gap-2 font-medium transition-colors"
                          >
                            <XCircle className="w-5 h-5" />
                            Reject
                          </button>
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                {analysisStatus === 'completed' && (
                  <div className="mt-6 space-y-4">
                    <div className="bg-gradient-to-r from-green-50 to-teal-50 border border-green-200 rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <CheckCircle className="w-6 h-6 text-green-600" />
                        <h3 className="font-semibold text-green-800">Analysis Complete!</h3>
                      </div>
                      <p className="text-sm text-green-700">All analysis steps have been completed successfully.</p>
                    </div>
                    
                    <div className="bg-white rounded-xl border border-slate-200 p-6 mb-4">
                      <h3 className="text-lg font-semibold text-slate-800 mb-4">Analysis Report Sections</h3>
                      <div className="space-y-2 mb-4">
                        {reportSections.analysis.map(section => (
                          <label key={section} className="flex items-center gap-2 cursor-pointer hover:bg-slate-50 p-2 rounded">
                            <input
                              type="checkbox"
                              checked={selectedSections.includes(section)}
                              onChange={() => toggleSection(section)}
                              className="w-4 h-4 text-teal-600 rounded focus:ring-2 focus:ring-teal-500"
                            />
                            <span className="text-slate-700">{section}</span>
                          </label>
                        ))}
                      </div>
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={customSection}
                          onChange={(e) => setCustomSection(e.target.value)}
                          placeholder="Add custom section"
                          className="flex-1 px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500"
                          onKeyPress={(e) => e.key === 'Enter' && addCustomSection()}
                        />
                        <button
                          onClick={addCustomSection}
                          disabled={!customSection.trim()}
                          className="px-4 py-2 bg-teal-100 text-teal-600 rounded-lg hover:bg-teal-200 disabled:opacity-50"
                        >
                          <PlusCircle className="w-5 h-5" />
                        </button>
                      </div>
                      {selectedSections.length > 0 && (
                        <div className="mt-3 flex flex-wrap gap-2">
                          {selectedSections.map(section => (
                            <span key={section} className="px-3 py-1 bg-teal-100 text-teal-700 rounded-full text-sm">
                              {section}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                    <button
                      onClick={() => generateReport('analysis')}
                      disabled={loading}
                      className="w-full px-6 py-4 bg-gradient-to-r from-teal-600 to-blue-600 text-white rounded-lg font-medium hover:from-teal-700 hover:to-blue-700 transition-all flex items-center justify-center gap-2 disabled:opacity-50"
                    >
                      <Download className="w-5 h-5" />
                      Generate Analysis Report
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ResearchAssistant;