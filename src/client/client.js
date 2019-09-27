import React, { useState } from 'react';
import logo from './google.jpg';
import './client.css';

function App() {
  /* Individual Component States */
  const [inputText, setInputText] = useState("");
  const [toxicity, setToxicity] = useState("N/A");
  const [isLoading, setLoading] = useState(false);
  /* Promise Functionality */
  function calculateToxicity() {
    const response = new Promise((resolve, reject) => {
      fetch('http://localhost:3001/toxicity', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          inputText: inputText,
        })
      }).then((response) => {
        const json = response.json().then((response) => {
          setToxicity(Number.parseFloat(json.toxicityScore).toFixed(3))
          resolve(response)
        })
      })
    })
    return response
  }
  /* Useful Functions */
  function submitToxicity() {
    if (isLoading === false) {
      setLoading(true);
      calculateToxicity().then((response) => {
        setLoading(false);
        setToxicity(response.toxicityScore)
      })
    }
  }
  function submitToxicitySecondary(e) {
    if (isLoading === false) {
      if (e.key === 'Enter') {
        setLoading(true);
        calculateToxicity().then((response) => {
          setLoading(false);
          setToxicity(response.toxicityScore)
        })
      }
    }
  }
  const active = (isLoading === true) ? "google-active" : ""
  const margin = (isLoading === true) ? "google-margin" : ""
  return (
    <div className="google-container">
      <div className="google-header">
        <div className="google-logo">
          <img className="google-image" src={logo} alt="logo" />
          <div className="google-subtext">Toxicity</div>
        </div>
      </div>
      <div className="google-secondary">
        <div className="google-searchbar">
          <div className="google-search-icon">
            <div className="google-search-icon-inner">
              <div className="search-icon">
                <svg focusable="false" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 25 25">
                  <path d="M15.5 14h-.79l-.28-.27A6.471 6.471 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z" />
                </svg>
              </div>
            </div>
          </div>
          <input className={["google-input", margin].join(' ')}
                 value={inputText}
                 onChange={e => setInputText(e.target.value)}
                 onKeyDown={e => submitToxicitySecondary(e)}/>
          <div className={["google-loader", active].join(' ')}/>
        </div>
      </div>
      <div className="google-secondary">
        <div className="google-scoring">
          {"Calculated Toxicity Score: " + toxicity}
        </div>
      </div>
      <div className="google-secondary">
        <div className="google-button-container">
          <div className="google-button"
                  onClick={submitToxicity}
                  onKeyDown={submitToxicity}>Calculate Toxicity</div>
          <a href="http://google.com" className="google-button">Return to Google</a>
        </div>
      </div>
    </div>
  );
}

export default App;
