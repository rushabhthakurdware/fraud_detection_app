import React, { useState, useEffect } from "react";

const API_BASE = "http://127.0.0.1:8000/api/v1";

export default function App() {
  const [amount, setAmount] = useState(1250);
  const [profile, setProfile] = useState("normal");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  // Generates 28 simulated PCA feature values based on selected profile pattern
  const getFeatures = (type) => {
    return Array.from({ length: 28 }, () =>
      type === "fraud" ? Math.random() * 4 + 2 : Math.random() * 0.5 - 0.25,
    );
  };

  // Fetches audit records stored in Supabase via FastAPI endpoint
  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/history`);
      const data = await res.json();
      setHistory(data);
    } catch (err) {
      console.error("Failed to retrieve audit history from Supabase", err);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const payload = {
      amount: parseFloat(amount),
      v_features: getFeatures(profile),
    };

    try {
      const res = await fetch(`${API_BASE}/assess-risk`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorData = await res.json();
        alert(
          `Backend Error (${res.status}): ${errorData.detail || "Internal Server Error"}`,
        );
        return;
      }

      const data = await res.json();
      setResult(data);
      fetchHistory();
    } catch (err) {
      console.error(err);
      alert(
        "Network Error: Could not connect to FastAPI server at " + API_BASE,
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-8 font-sans">
      <div className="max-w-5xl mx-auto space-y-8">
        <header className="border-b border-slate-800 pb-4">
          <h1 className="text-3xl font-bold text-indigo-400">
            Enterprise Fraud Risk Dashboard
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            React + FastAPI + Supabase Postgres Database Architecture
          </p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Controls */}
          <form
            onSubmit={handleSubmit}
            className="bg-slate-800 p-6 rounded-xl border border-slate-700 space-y-4 shadow-lg"
          >
            <h2 className="text-xl font-semibold text-slate-200">
              Transaction Simulator
            </h2>
            <div>
              <label className="block text-sm font-medium text-slate-400 mb-1">
                Transaction Amount ($)
              </label>
              <input
                type="number"
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-white outline-none focus:border-indigo-500"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-400 mb-1">
                Behavior Pattern Profile
              </label>
              <select
                value={profile}
                onChange={(e) => setProfile(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-white outline-none focus:border-indigo-500"
              >
                <option value="normal">Legitimate Behavior Pattern</option>
                <option value="fraud">Fraud / High Outlier Pattern</option>
              </select>
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-indigo-600 hover:bg-indigo-500 font-semibold py-2 rounded transition"
            >
              {loading
                ? "Calculating Risk Score..."
                : "Evaluate Transaction Risk"}
            </button>
          </form>

          {/* Current Output */}
          <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 flex flex-col justify-between shadow-lg">
            <div>
              <h2 className="text-xl font-semibold mb-4 text-slate-200">
                ML Model Output
              </h2>
              {result ? (
                <div className="space-y-4">
                  <div
                    className={`inline-block px-3 py-1 rounded-full text-xs font-bold ${result.flagged ? "bg-red-600 text-white" : "bg-green-600 text-white"}`}
                  >
                    {result.flagged
                      ? "FLAGGED: HIGH RISK"
                      : "CLEARED: LOW RISK"}
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm text-slate-400">
                      XGBoost Risk Score:{" "}
                      <span className="font-bold text-white">
                        {(result.risk_score * 100).toFixed(2)}%
                      </span>
                    </p>
                    <p className="text-sm text-slate-400">
                      Isolation Forest Anomaly:{" "}
                      <span className="font-bold text-white">
                        {result.is_anomaly ? "ANOMALY DETECTED" : "NORMAL"}
                      </span>
                    </p>
                  </div>
                  {/* Safe rendering for reasons array */}
                  <div>
                    <h3 className="text-sm font-semibold text-slate-300 mb-1">
                      Explainability Risk Drivers:
                    </h3>
                    <ul className="list-disc list-inside text-xs text-slate-400 space-y-1">
                      {result.reasons && Array.isArray(result.reasons) ? (
                        result.reasons.map((r, idx) => <li key={idx}>{r}</li>)
                      ) : (
                        <li>No breakdown available.</li>
                      )}
                    </ul>
                  </div>
                </div>
              ) : (
                <p className="text-slate-500 text-sm">
                  Submit a transaction from the left panel to trigger ML
                  inference.
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Audit Logs fetched directly from Supabase via FastAPI */}
        <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-lg">
          <h2 className="text-xl font-semibold text-slate-200 mb-4">
            Audit History Log (Saved in Supabase Postgres)
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-slate-400">
              <thead className="bg-slate-900 text-slate-300">
                <tr>
                  <th className="p-3">Amount</th>
                  <th className="p-3">Risk Score</th>
                  <th className="p-3">Anomaly</th>
                  <th className="p-3">Status</th>
                </tr>
              </thead>
              <tbody>
                {history.length > 0 ? (
                  history.map((row) => (
                    <tr
                      key={row.id}
                      className="border-b border-slate-700 hover:bg-slate-700/50"
                    >
                      <td className="p-3 text-white">${row.amount}</td>
                      <td className="p-3">
                        {(row.risk_score * 100).toFixed(2)}%
                      </td>
                      <td className="p-3">{row.is_anomaly ? "Yes" : "No"}</td>
                      <td className="p-3 font-semibold">
                        <span
                          className={
                            row.flagged ? "text-red-400" : "text-green-400"
                          }
                        >
                          {row.flagged ? "Flagged" : "Cleared"}
                        </span>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="4" className="p-4 text-center text-slate-500">
                      No records saved yet in Supabase. Submit a transaction
                      above!
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
