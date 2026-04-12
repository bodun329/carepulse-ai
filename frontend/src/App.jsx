import { useState } from "react";
import axios from "axios";

export default function App() {
  const [symptoms, setSymptoms] = useState("");
  const [patients, setPatients] = useState(() => {
    return JSON.parse(localStorage.getItem("patients")) || [];
  });

  const analyze = async () => {
    const res = await axios.post("http://127.0.0.1:8000/predict", {
      text: symptoms,
    });

    const newPatient = {
      id: Date.now(),
      text: symptoms,
      risk: res.data.risk,
      score: res.data.score,
    };

    const updated = [newPatient, ...patients];
    setPatients(updated);
    localStorage.setItem("patients", JSON.stringify(updated));

    setSymptoms("");
  };

  const riskTrend = {
    HIGH: patients.filter(p => p.risk === "HIGH").length,
    MEDIUM: patients.filter(p => p.risk === "MEDIUM").length,
    LOW: patients.filter(p => p.risk === "LOW").length,
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <h1 className="text-4xl font-bold text-center text-blue-700 mb-6">
        🏥 CarePulse AI Dashboard
      </h1>

      {/* STATS */}
      <div className="grid grid-cols-3 gap-4 max-w-4xl mx-auto mb-6">
        <div className="bg-red-100 p-4 rounded text-center">
          <h2 className="text-xl font-bold">{riskTrend.HIGH}</h2>
          <p>High Risk</p>
        </div>
        <div className="bg-yellow-100 p-4 rounded text-center">
          <h2 className="text-xl font-bold">{riskTrend.MEDIUM}</h2>
          <p>Medium Risk</p>
        </div>
        <div className="bg-green-100 p-4 rounded text-center">
          <h2 className="text-xl font-bold">{riskTrend.LOW}</h2>
          <p>Low Risk</p>
        </div>
      </div>

      {/* INPUT */}
      <div className="max-w-xl mx-auto bg-white p-6 rounded shadow mb-6">
        <textarea
          className="w-full border p-3 rounded mb-4"
          placeholder="Enter patient symptoms..."
          value={symptoms}
          onChange={(e) => setSymptoms(e.target.value)}
        />

        <button
          onClick={analyze}
          className="w-full bg-blue-600 text-white py-2 rounded"
        >
          Analyze Patient
        </button>
      </div>

      {/* PATIENT LIST */}
      <div className="max-w-3xl mx-auto">
        {patients.map((p) => (
          <div
            key={p.id}
            className={`p-4 mb-3 rounded shadow ${
              p.risk === "HIGH"
                ? "bg-red-100"
                : p.risk === "MEDIUM"
                ? "bg-yellow-100"
                : "bg-green-100"
            }`}
          >
            <p><b>Symptoms:</b> {p.text}</p>
            <p><b>Risk:</b> {p.risk}</p>
            <p><b>Score:</b> {(p.score * 100).toFixed(1)}%</p>
          </div>
        ))}
      </div>
    </div>
  );
}