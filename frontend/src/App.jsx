import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './App.css';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  LineChart, Line, Legend
} from 'recharts';
import { Bed, AlertTriangle, CheckCircle, Calendar, Loader2 } from 'lucide-react';

const App = () => {
  const [date, setDate] = useState(new Date().toISOString().split('T')[0]);
  const [data, setData] = useState(null); // Now holds an array of 7 days
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchPrediction = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // Connects to your Python FastAPI backend
      const response = await axios.get(`http://localhost:8000/predict?date=${date}`);

      // Check if the backend returned an error logic (though usually 200 OK means fine, check data content)
      if (response.data.error) {
        setError(response.data.error);
        setData(null);
      } else {
        // Response is now a list of 7 days
        setData(response.data);
      }
    } catch (err) {
      console.error("Error fetching data:", err);
      // Simplify error message for user
      setError("Failed to connect to the backend. Is it running?");
      setData(null);
    }
    setLoading(false);
  }, [date]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchPrediction();
  }, [fetchPrediction]);

  // Derive current day data for summary cards (first item in array)
  const currentDay = data && data.length > 0 ? data[0] : null;

  // Chart data for BarChart (Single Day)
  const barChartData = currentDay ? [
    { name: 'Predicted', value: currentDay.predicted_occupancy },
    { name: 'Worst Case', value: currentDay.worst_case }
  ] : [];

  return (
    <div className="app-container">

      {/* HEADER SECTION */}
      <header className="header">
        <div className="icon-wrapper">
          <Bed size={24} color="white" />
        </div>
        <h1 className="title">Hospital Bed Occupancy (Project 62A)</h1>
      </header>

      {/* CONTROLS SECTION */}
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

      {/* DATA DISPLAY SECTION */}
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
          <button className="retry-btn" onClick={fetchPrediction}>
            Retry
          </button>
        </div>
      ) : currentDay ? (
        <div className="dashboard-grid">

          {/* LEFT COLUMN: STATUS CARDS */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <div className="card">
              <p className="card-label">Available Beds ({currentDay.date})</p>
              <h2 className="stat-value">{currentDay.available_beds}</h2>
            </div>

            <div className={`status-card ${currentDay.risk === 'LOW' ? 'status-low' : 'status-critical'}`}>
              {currentDay.risk === 'LOW' ? <CheckCircle size={24} /> : <AlertTriangle size={24} />}
              <span>RISK STATUS: {currentDay.risk}</span>
            </div>

            {/* BAR CHART (Single Day Context) */}
            <div className="card">
              <p className="card-label" style={{ marginBottom: '24px' }}>Occupancy Forecast vs Capacity ({currentDay.date})</p>
              <div className="chart-container">
                <ResponsiveContainer>
                  <BarChart data={barChartData}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                    <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: 'var(--text-muted)' }} />
                    <YAxis domain={[0, 150]} axisLine={false} tickLine={false} tick={{ fill: 'var(--text-muted)' }} />
                    <Tooltip
                      cursor={{ fill: 'transparent' }}
                      contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                    />
                    <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                      <Cell fill="var(--primary)" />
                      <Cell fill="var(--danger)" />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN: 7-DAY TREND CHART */}
          <div className="card" style={{ gridColumn: '1 / -1' }}> {/* Span full width if grid allows, or just be another card */}
            <p className="card-label" style={{ marginBottom: '24px' }}>7-Day Occupancy Trend</p>
            <div className="chart-container" style={{ height: '400px' }}>
              <ResponsiveContainer>
                <LineChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                  <XAxis
                    dataKey="date"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: 'var(--text-muted)' }}
                    tickFormatter={(value) => value.split('-').slice(1).join('/')} // Show MM/DD
                  />
                  <YAxis domain={[0, 150]} axisLine={false} tickLine={false} tick={{ fill: 'var(--text-muted)' }} />
                  <Tooltip
                    contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                  />
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
  );
};

export default App;