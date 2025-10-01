import React, { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [sheetName, setSheetName] = useState('');
  const [message, setMessage] = useState('');
  const [downloadUrl, setDownloadUrl] = useState('');
  const [predictions, setPredictions] = useState([]);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setMessage('');
    setDownloadUrl('');
    setPredictions([]);
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage('Please choose an Excel file to upload.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('sheet_name', sheetName);

    setMessage('Uploading...');

    try {
      // --- Excel download request ---
      const resp = await fetch('/', {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        const txt = await resp.json();
        setMessage('Server error: ' + (txt.error || resp.statusText));
        return;
      }

      const blob = await resp.blob();
      const url = window.URL.createObjectURL(blob);
      setDownloadUrl(url);
      setMessage('Prediction complete!');

      // --- JSON display request ---
      const jsonResp = await fetch('/', {
        method: 'POST',
        body: formData,
      });

      if (jsonResp.ok) {
        const preds = await jsonResp.json();
        setPredictions(preds);
      } else {
        setPredictions([]);
      }
    } catch (err) {
      setMessage('Upload failed: ' + err.message);
    }
  };

  return (
    <div className="app-container">
      <img src="/mepclogo.png" alt="Logo" className="logo pt-4" />
      <h1 className="subtitle is-1 pt-3">WESM Forecasting Tool</h1>
      <p className="subtitle is-7 p-1 pl-6 pr-6 has-text-centered">
        "The WESM Forecasting Tool is a data-driven application designed to predict market prices and demand in the <code>Wholesale Electricity Spot Market.</code> It enables users to upload datasets, analyze trends, and generate accurate forecasts to support better decision-making for energy trading, planning, and risk management."
      </p>
      <div className="columns is-centered p-4">
        <div className="is-justify-content-left">
          <div className="input-group">
            <label className="subtitle is-7 is-italic">Excel Upload<i></i></label>
          </div>
          <input type="file" accept=".xlsx,.xls" onChange={handleFileChange} className="mb-1 p-3" />
          <div className="input-group">
            <label className="subtitle is-7 is-italic">Sheet Name<i></i></label>
            <input
              className="input is-small p-3"
              type="text"
              placeholder="Example: Sheet1"
              value={sheetName}
              onChange={(e) => setSheetName(e.target.value)}
            />
          </div>
          <div className="button-group columns is-centered">
            <button onClick={handleUpload} disabled={!file}>Generate</button>
          </div>
        </div>
      </div>

      <div className="is">
        {predictions.length > 0 && (
          <div className="results">
            <hr className="m-6" />
            <div className="in-line-flex is-justify-content-space-between is-align-items-center mb-4">
              <h1>Result</h1>
              <div className="is-flex is-justify-content-flex-end">
                <div className="is-warning mr-2">
                  {downloadUrl && (
                    <a
                      href={downloadUrl}
                      download="predicted_output.xlsx"
                      className="is-centered"
                      style={{ textDecoration: 'none' }}
                    >
                      <button>
                        <i className="fas fa-download pt-1 pb-1"></i>
                      </button>
                    </a>
                  )}
                </div>
                <div className="has-text-centered">
                  <button className="is-danger" onClick={() => window.location.reload()}>
                    <i className="fas fa-rotate pt-1 pb-1"></i>
                  </button>
                </div>
              </div>
            </div>
            <table className="prediction-table">
              <thead>
                <tr>
                  <th>Hour</th>
                  <th>Predicted EOD WESM Price</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((p, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>{p.toFixed ? p.toFixed(4) : p}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
