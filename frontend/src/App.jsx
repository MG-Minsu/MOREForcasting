import React, { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [sheetName, setSheetName] = useState('');
  const [message, setMessage] = useState('');
  const [downloadUrl, setDownloadUrl] = useState('');
  const [tableData, setTableData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [stats, setStats] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setMessage('');
    setDownloadUrl('');
    setTableData([]);
    setColumns([]);
    setStats(null);
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
      // --- JSON display request (do this first to get data) ---
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
        // Set table data
        setTableData(result.data);
        setColumns(result.columns);
        setStats(result.prediction_stats);
        setMessage('Prediction complete!');
      } else {
        setMessage('Error: ' + result.error);
        return;
      }

      // --- Excel download request (create new FormData) ---
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
            <label htmlFor="" className="subtitle is-7 is-italic">Excel Upload</label>
          </div>
          <input 
            type="file" 
            accept=".xlsx,.xls" 
            onChange={handleFileChange} 
            className="mb-1 p-3"
          />
          
          <div className="input-group">
            <label htmlFor="" className="subtitle is-7 is-italic">Sheet Name (optional)</label>
            <input
              className="input is-small p-3"
              type="text"
              placeholder="Example: Sheet1 (leave blank for first sheet)"
              value={sheetName}
              onChange={(e) => setSheetName(e.target.value)}
            />
          </div>
          
          <div className="button-group columns is-centered">
            <button onClick={handleUpload} disabled={!file}>Generate</button>
          </div>
          
          {message && (
            <p className="has-text-centered mt-2" style={{
              color: message.includes('error') || message.includes('failed') ? 'red' : 'green'
            }}>
              {message}
            </p>
          )}
        </div>
        {/* End */}
      </div>

      <div className="is">
        {tableData.length > 0 && (
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

            {/* Statistics Summary */}
            {stats && (
              <div className="box mb-4" style={{background: '#f0f8ff', padding: '15px'}}>
                <h3 className="subtitle is-5">📊 Prediction Statistics</h3>
                <div className="columns">
                  <div className="column">
                    <strong>Min:</strong> {stats.min.toFixed(2)}
                  </div>
                  <div className="column">
                    <strong>Max:</strong> {stats.max.toFixed(2)}
                  </div>
                  <div className="column">
                    <strong>Mean:</strong> {stats.mean.toFixed(2)}
                  </div>
                  <div className="column">
                    <strong>Median:</strong> {stats.median.toFixed(2)}
                  </div>
                </div>
              </div>
            )}

            {/* Full Data Table */}
            <div style={{overflowX: 'auto'}}>
              <table className="prediction-table">
                <thead>
                  <tr>
                    <th>#</th>
                    {columns.map((col, idx) => (
                      <th 
                        key={idx}
                        style={{
                          background: col === 'Predicted_EOD_WESM_Price' ? '#4CAF50' : undefined,
                          color: col === 'Predicted_EOD_WESM_Price' ? 'white' : undefined
                        }}
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tableData.map((row, rowIdx) => (
                    <tr key={rowIdx}>
                      <td>{rowIdx + 1}</td>
                      {columns.map((col, colIdx) => (
                        <td 
                          key={colIdx}
                          style={{
                            background: col === 'Predicted_EOD_WESM_Price' ? '#e8f5e9' : undefined,
                            fontWeight: col === 'Predicted_EOD_WESM_Price' ? 'bold' : undefined
                          }}
                        >
                          {row[col] !== null && row[col] !== undefined
                            ? (typeof row[col] === 'number' 
                                ? row[col].toFixed(4) 
                                : row[col])
                            : '-'
                          }
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
