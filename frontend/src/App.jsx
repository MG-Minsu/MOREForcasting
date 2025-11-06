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
    if (sheetName) {
      formData.append('sheet_name', sheetName);
    }

    setMessage('Uploading...');

    try {
      // --- JSON display request ---
      const jsonResp = await fetch('https://moreforcasting.onrender.com/predict-json', {
        method: 'POST',
        body: formData,
      });

      if (!jsonResp.ok) {
        const errorData = await jsonResp.json();
        setMessage('Server error: ' + (errorData.error || jsonResp.statusText));
        return;
      }

      const result = await jsonResp.json();

      if (result.success) {
        // Extract only the Predicted_EOD_WESM_Price values
        const predictionValues = result.data.map(row => row.Predicted_EOD_WESM_Price);
        setPredictions(predictionValues);
        setMessage('Prediction complete!');
      } else {
        setMessage('Error: ' + result.error);
        return;
      }

      // --- Excel download request ---
      const formDataForFile = new FormData();
      formDataForFile.append('file', file);
      if (sheetName) {
        formDataForFile.append('sheet_name', sheetName);
      }

      const resp = await fetch('https://moreforcasting.onrender.com/predict-file', {
        method: 'POST',
        body: formDataForFile,
      });

      if (resp.ok) {
        const blob = await resp.blob();
        const url = window.URL.createObjectURL(blob);
        setDownloadUrl(url);
      }

    } catch (err) {
      setMessage('Upload failed: ' + err.message);
      console.error('Error:', err);
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
        <div>
          {/* Placeholder */}
        </div>
        
        {/* Form Field */}
        <div className="is-justify-content-left">
          <div className="input-group">
            <label htmlFor="" className="subtitle is-7 is-italic">Excel Upload<i></i></label>
          </div>
          <input 
            type="file" 
            accept=".xlsx,.xls" 
            onChange={handleFileChange} 
            className="mb-1 p-3"
          />
          
          <div className="input-group">
            <label htmlFor="" className="subtitle is-7 is-italic">Sheet Name<i></i></label>
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
          
          {message && (
            <p className="has-text-centered mt-2" style={{
              color: message.includes('error') || message.includes('failed') ? 'red' : 
                     message.includes('complete') ? 'green' : 'blue'
            }}>
              {message}
            </p>
          )}
        </div>
        {/* End */}
      </div>

      <div className="is">
        {predictions.length > 0 && (
          <div className="results">
            <hr className="m-6"/>
            
            <div className="in-line-flex is-justify-content-space-between is-align-items-center mb-4">
              <h1>Result</h1>
              
              <div className="is-flex is-justify-content-flex-end">
                {downloadUrl && (
                  <div className="is-warning mr-2">
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
                  </div>
                )}
                
                <div className="has-text-centered">
                  <button
                    className="is-danger"
                    onClick={() => window.location.reload()}
                  >
                    <i className="fas fa-rotate pt-1 pb-1"></i>
                  </button>
                </div>
              </div>
            </div>

            {/* Simple table showing only Hour and Predicted Price */}
            <table className="prediction-table">
              <thead>
                <tr>
                  <th>Hour</th>
                  <th>Predicted EOD WESM Price</th>
                </tr>
              </thead>
              <tbody>
                {predictions.slice(0, 24).map((p, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>
                      {p !== null && p !== undefined
                        ? (typeof p === 'number' ? p.toFixed(4) : p)
                        : '-'
                      }
                    </td>
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
