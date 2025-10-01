import React, { useState } from "react";

function PredictionModal({ predictions, downloadUrl }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div>
      {/* Open Modal Button */}
      <button className="button is-primary" onClick={() => setIsOpen(true)}>
        Show Predictions
      </button>

      {/* Modal */}
      {isOpen && (
        <div className="modal is-active">
          <div
            className="modal-background"
            onClick={() => setIsOpen(false)}
          ></div>

          <div className="modal-card">
            <header className="modal-card-head">
              <p className="modal-card-title">Predicted Values</p>
              <button
                className="delete"
                aria-label="close"
                onClick={() => setIsOpen(false)}
              ></button>
            </header>

            <section className="modal-card-body">
              {predictions.length > 0 && (
                <table className="table is-fullwidth">
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
              )}

              {downloadUrl && (
                <div className="mt-4 has-text-centered">
                  <a
                    href={downloadUrl}
                    download="predicted_output.xlsx"
                    className="button is-success"
                  >
                    Download Excel
                  </a>
                </div>
              )}
            </section>
          </div>
        </div>
      )}
    </div>
  );
}

export default PredictionModal;
