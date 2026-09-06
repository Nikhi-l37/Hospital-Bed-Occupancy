import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './App.css';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  LineChart, Line, Legend
} from 'recharts';
import {
  Bed, AlertTriangle, CheckCircle, Calendar, Loader2,
  Pill, Clock, Activity, Upload, FileText, Search, Stethoscope,
  HeartPulse, Sparkles, ChevronRight
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

const App = () => {
  // ─── Tab state ──────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState('dashboard');

  // ─── Bed Occupancy state (existing) ─────────────────────────
  const [date, setDate] = useState(new Date().toISOString().split('T')[0]);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // ─── Prescription OCR state ─────────────────────────────────
  const [prescriptionFile, setPrescriptionFile] = useState(null);
  const [ocrLoading, setOcrLoading] = useState(false);
  const [ocrResult, setOcrResult] = useState(null);
  const [ocrError, setOcrError] = useState(null);

  // ─── Symptom Analyzer state ─────────────────────────────────
  const [symptomText, setSymptomText] = useState('');
  const [symptomLoading, setSymptomLoading] = useState(false);
  const [symptomResult, setSymptomResult] = useState(null);
  const [symptomError, setSymptomError] = useState(null);

  // ─── Bed Occupancy fetch ────────────────────────────────────
  const fetchPrediction = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_BASE}/predict?date=${date}`);
      if (response.data.error) {
        setError(response.data.error);
        setData(null);
      } else {
        setData(response.data);
      }
    } catch (err) {
      console.error("Error fetching data:", err);
      setError("Failed to connect to the backend. Is it running?");
      setData(null);
    }
    setLoading(false);
  }, [date]);

  useEffect(() => {
    if (activeTab === 'dashboard') {
      fetchPrediction();
    }
  }, [fetchPrediction, activeTab]);

  // ─── Scan Prescription ──────────────────────────────────────
  const handleScanPrescription = async () => {
    if (!prescriptionFile) return;
    setOcrLoading(true);
    setOcrError(null);
    setOcrResult(null);
    try {
      const formData = new FormData();
      formData.append('file', prescriptionFile);
      const response = await axios.post(`${API_BASE}/upload-prescription`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setOcrResult(response.data.extracted_text);
    } catch (err) {
      console.error("OCR Error:", err);
      setOcrError(err.response?.data?.detail || "Failed to scan prescription.");
    }
    setOcrLoading(false);
  };

  // ─── Analyze Symptoms ───────────────────────────────────────
  const handleAnalyzeSymptoms = async () => {
    if (!symptomText.trim()) return;
    setSymptomLoading(true);
    setSymptomError(null);
    setSymptomResult(null);
    try {
      // Try AI-powered analysis first
      const response = await axios.post(`${API_BASE}/ai-analyze`, {
        patient_text: symptomText
      });
      setSymptomResult(response.data);
    } catch (err) {
      console.warn("AI analysis failed, falling back to keyword matching:", err);
      // Fallback to keyword-based analysis
      try {
        const fallback = await axios.post(`${API_BASE}/analyze-symptoms`, {
          patient_text: symptomText
        });
        setSymptomResult({ ...fallback.data, ai_powered: false });
      } catch (err2) {
        console.error("Symptom Error:", err2);
        setSymptomError(err2.response?.data?.detail || "Failed to analyze symptoms.");
      }
    }
    setSymptomLoading(false);
  };

  // ─── Derived data ──────────────────────────────────────────
  const currentDay = data && data.length > 0 ? data[0] : null;
  const barChartData = currentDay ? [
    { name: 'Predicted', value: currentDay.predicted_occupancy },
    { name: 'Worst Case', value: currentDay.worst_case }
  ] : [];

  // ═══════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════
  return (
    <div className="app-container">

      {/* ── HEADER ────────────────────────────────────────── */}
      <header className="header">
        <div className="icon-wrapper">
          <HeartPulse size={24} color="white" />
        </div>
        <div>
          <h1 className="title">Smart Hospital System</h1>
          <p className="subtitle">Project  — AI-Powered Patient Care</p>
        </div>
      </header>

      {/* ── TAB NAVIGATION ────────────────────────────────── */}
      <nav className="tab-nav">
        <button
          className={`tab-btn ${activeTab === 'dashboard' ? 'tab-active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          <Bed size={18} />
          <span>Bed Occupancy</span>
        </button>
        <button
          className={`tab-btn ${activeTab === 'patient' ? 'tab-active' : ''}`}
          onClick={() => setActiveTab('patient')}
        >
          <Stethoscope size={18} />
          <span>Patient Dashboard</span>
        </button>
      </nav>

      {/* ═══════════════════════════════════════════════════ */}
      {/* TAB: BED OCCUPANCY (existing)                      */}
      {/* ═══════════════════════════════════════════════════ */}
      {activeTab === 'dashboard' && (
        <div className="tab-content fade-in">
          {/* Controls */}
          <div className="controls">
            <Calendar size={20} color="var(--primary)" />
            <input
              type="date"
              className="date-input"
              value={date}
              onChange={(e) => setDate(e.target.value)}
            />
            {loading && <Loader2 className="animate-spin" size={20} color="var(--primary)" />}
          </div>

          {loading ? (
            <div className="state-container">
              <Loader2 className="animate-spin" size={40} style={{ marginBottom: '16px', color: 'var(--primary)' }} />
              <p>Analyzing Hospital Data...</p>
            </div>
          ) : error ? (
            <div className="state-container" style={{ borderColor: 'var(--danger)', color: 'var(--danger)' }}>
              <AlertTriangle size={40} style={{ marginBottom: '16px' }} />
              <p style={{ fontWeight: 'bold' }}>Error Loading Data</p>
              <p>{error}</p>
              <button className="btn-primary" onClick={fetchPrediction} style={{ marginTop: 16 }}>Retry</button>
            </div>
          ) : currentDay ? (
            <div className="dashboard-grid">
              <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                <div className="card">
                  <p className="card-label">Available Beds ({currentDay.date})</p>
                  <h2 className="stat-value">{currentDay.available_beds}</h2>
                </div>
                <div className={`status-card ${currentDay.risk === 'LOW' ? 'status-low' : 'status-critical'}`}>
                  {currentDay.risk === 'LOW' ? <CheckCircle size={24} /> : <AlertTriangle size={24} />}
                  <span>RISK STATUS: {currentDay.risk}</span>
                </div>
                <div className="card">
                  <p className="card-label" style={{ marginBottom: '24px' }}>Occupancy Forecast ({currentDay.date})</p>
                  <div className="chart-container">
                    <ResponsiveContainer>
                      <BarChart data={barChartData}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                        <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: 'var(--text-muted)' }} />
                        <YAxis domain={[0, 150]} axisLine={false} tickLine={false} tick={{ fill: 'var(--text-muted)' }} />
                        <Tooltip cursor={{ fill: 'transparent' }} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                        <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                          <Cell fill="var(--primary)" />
                          <Cell fill="var(--danger)" />
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
              <div className="card" style={{ gridColumn: '1 / -1' }}>
                <p className="card-label" style={{ marginBottom: '24px' }}>7-Day Occupancy Trend</p>
                <div className="chart-container" style={{ height: '400px' }}>
                  <ResponsiveContainer>
                    <LineChart data={data}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                      <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{ fill: 'var(--text-muted)' }} tickFormatter={(v) => v.split('-').slice(1).join('/')} />
                      <YAxis domain={[0, 150]} axisLine={false} tickLine={false} tick={{ fill: 'var(--text-muted)' }} />
                      <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                      <Legend />
                      <Line type="monotone" dataKey="predicted_occupancy" name="Predicted" stroke="var(--primary)" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                      <Line type="monotone" dataKey="worst_case" name="Worst Case" stroke="var(--danger)" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          ) : (
            <div className="state-container">
              <p>Please select a date or ensure the backend server is running.</p>
            </div>
          )}
        </div>
      )}

      {/* ═══════════════════════════════════════════════════ */}
      {/* TAB: PATIENT DASHBOARD (new Modules 4 & 5)         */}
      {/* ═══════════════════════════════════════════════════ */}
      {activeTab === 'patient' && (
        <div className="tab-content fade-in">
          <div className="patient-grid">

            {/* ── LEFT: Prescription Upload ──────────────── */}
            <div className="card patient-card">
              <div className="card-header-row">
                <div className="card-icon upload-icon">
                  <Upload size={20} />
                </div>
                <div>
                  <h3 className="card-title">Scan Prescription</h3>
                  <p className="card-desc">Upload an image to extract text via OCR</p>
                </div>
              </div>

              <label className="file-drop-zone" htmlFor="prescription-upload">
                <input
                  id="prescription-upload"
                  type="file"
                  accept="image/*"
                  className="file-input-hidden"
                  onChange={(e) => {
                    setPrescriptionFile(e.target.files[0]);
                    setOcrResult(null);
                    setOcrError(null);
                  }}
                />
                {prescriptionFile ? (
                  <div className="file-selected">
                    <FileText size={28} color="var(--primary)" />
                    <span className="file-name">{prescriptionFile.name}</span>
                    <span className="file-size">({(prescriptionFile.size / 1024).toFixed(1)} KB)</span>
                  </div>
                ) : (
                  <div className="file-placeholder">
                    <Upload size={32} color="var(--text-muted)" />
                    <p>Click or drop a prescription image here</p>
                    <span className="file-hint">JPG, PNG, WEBP supported</span>
                  </div>
                )}
              </label>

              <button
                className="btn-primary btn-full"
                onClick={handleScanPrescription}
                disabled={!prescriptionFile || ocrLoading}
              >
                {ocrLoading ? (
                  <><Loader2 className="animate-spin" size={18} /> Scanning...</>
                ) : (
                  <><Search size={18} /> Scan Prescription</>
                )}
              </button>

              {/* OCR Results */}
              {ocrError && (
                <div className="result-error">
                  <AlertTriangle size={16} /> {ocrError}
                </div>
              )}
              {ocrResult && (
                <div className="ocr-result-box">
                  <p className="result-label"><FileText size={14} /> Extracted Text</p>
                  <div className="ocr-text-list">
                    {ocrResult.map((text, idx) => (
                      <div key={idx} className="ocr-text-item">
                        <ChevronRight size={14} color="var(--primary)" />
                        <span>{text}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* ── RIGHT: Symptom Analyzer ────────────────── */}
            <div className="card patient-card">
              <div className="card-header-row">
                <div className="card-icon symptom-icon">
                  <Stethoscope size={20} />
                </div>
                <div>
                  <h3 className="card-title">Symptom Analyzer</h3>
                  <p className="card-desc">Describe your symptoms to get recommendations</p>
                </div>
              </div>

              <textarea
                className="symptom-textarea"
                placeholder="e.g. I have been having a severe headache and fever since yesterday. I also have a sore throat and cough..."
                value={symptomText}
                onChange={(e) => setSymptomText(e.target.value)}
                rows={6}
              />

              <button
                className="btn-accent btn-full"
                onClick={handleAnalyzeSymptoms}
                disabled={!symptomText.trim() || symptomLoading}
              >
                {symptomLoading ? (
                  <><Loader2 className="animate-spin" size={18} /> Analyzing...</>
                ) : (
                  <><Activity size={18} /> Get Recommendations</>
                )}
              </button>

              {symptomError && (
                <div className="result-error">
                  <AlertTriangle size={16} /> {symptomError}
                </div>
              )}
            </div>
          </div>

          {/* ══ TREATMENT VISUALIZATION ════════════════════ */}
          {symptomResult && (
            <div className="treatment-section fade-in">
              <div className="section-header">
                <Sparkles size={22} color="var(--accent)" />
                <h2 className="section-title">Treatment Recommendations</h2>
                {symptomResult.ai_powered && (
                  <span className="ai-badge">✨ AI Powered</span>
                )}
              </div>

              {/* AI Summary */}
              {symptomResult.general_advice && (
                <div className={`ai-summary-box ${symptomResult.see_doctor ? 'ai-summary-doctor' : ''}`}>
                  <Stethoscope size={20} />
                  <p>{symptomResult.general_advice}</p>
                </div>
              )}

              {symptomResult.detected_symptoms.length > 0 && (
                <>
                  {/* Detected Symptoms Pills */}
                  <div className="symptom-pills">
                    {symptomResult.detected_symptoms.map((s, i) => (
                      <span key={i} className="symptom-pill">
                        <Activity size={14} />
                        {s.charAt(0).toUpperCase() + s.slice(1)}
                      </span>
                    ))}
                  </div>

                  {/* Recommendation Cards */}
                  <div className="rec-cards-grid">
                    {symptomResult.recommendations.map((rec, i) => (
                      <div key={i} className={`rec-card ${rec.urgency === 'severe' ? 'rec-card-severe' : rec.urgency === 'moderate' ? 'rec-card-moderate' : ''}`}>
                        <div className="rec-card-header">
                          <span className="rec-symptom-badge">
                            {rec.symptom.charAt(0).toUpperCase() + rec.symptom.slice(1)}
                          </span>
                          <span className={`urgency-badge urgency-${rec.urgency}`}>
                            {rec.urgency === 'severe' ? '🔴' : rec.urgency === 'moderate' ? '🟡' : '🟢'} {rec.urgency.toUpperCase()}
                          </span>
                        </div>
                        <div className="rec-card-body">
                          <div className="rec-row">
                            <Pill size={18} className="rec-icon pill-icon" />
                            <div>
                              <p className="rec-label">Medicine</p>
                              <p className="rec-value">{rec.medicine}</p>
                            </div>
                          </div>
                          <div className="rec-divider" />
                          <div className="rec-row">
                            <Clock size={18} className="rec-icon clock-icon" />
                            <div>
                              <p className="rec-label">Schedule</p>
                              <p className="rec-value">{rec.timeline}</p>
                            </div>
                          </div>
                          {rec.advice && (
                            <>
                              <div className="rec-divider" />
                              <div className="rec-row">
                                <Stethoscope size={18} className="rec-icon advice-icon" />
                                <div>
                                  <p className="rec-label">Doctor's Advice</p>
                                  <p className="rec-value rec-advice">{rec.advice}</p>
                                </div>
                              </div>
                            </>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default App;