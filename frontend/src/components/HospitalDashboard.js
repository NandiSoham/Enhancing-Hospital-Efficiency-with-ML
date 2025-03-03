import React, { useState, useEffect } from 'react';
import axios from 'axios';

function HospitalDashboard() {
  // Define state variables
  const [patientData, setPatientData] = useState({
    hospitalType: '',
    department: '',
    wardType: '',
    admissionType: '',
    severity: '',
    age: '',
    bedGrade: '',
    cityCode: '',
    visitors: '',
    admissionDeposit: ''
  });
  const [prediction, setPrediction] = useState(null);
  const [predictionMade, setPredictionMade] = useState(false);
  const [stayDistribution, setStayDistribution] = useState([]);

  // Handle input changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setPatientData({
      ...patientData,
      [name]: value
    });
  };

  // Make prediction request to backend
  const makePrediction = async () => {
    try {
      const response = await axios.post('http://localhost:5000/predict', {
        hospital_type_code: patientData.hospitalType,
        department: patientData.department,
        ward_type: patientData.wardType,
        type_of_admission: patientData.admissionType,
        severity_of_illness: patientData.severity,
        age: patientData.age,
        bed_grade: patientData.bedGrade,
        city_code_patient: patientData.cityCode,
        visitors_with_patient: patientData.visitors,
        admission_deposit: patientData.admissionDeposit
      });
      
      setPrediction(response.data.prediction);
      setPredictionMade(true);
    } catch (error) {
      console.error('Error making prediction:', error);
    }
  };

  // Fetch stay distribution stats on component mount
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await axios.get('http://localhost:5000/stats');
        setStayDistribution(response.data.distribution);
      } catch (error) {
        console.error('Error fetching stats:', error);
      }
    };
    
    fetchStats();
  }, []);

  return (
    <div className="dashboard-container">
      <h1>Hospital Stay Prediction Dashboard</h1>
      
      <div className="input-section">
        <h2>Patient Information</h2>
        <div className="form-group">
          <label htmlFor="hospitalType">Hospital Type:</label>
          <input 
            type="text" 
            id="hospitalType" 
            name="hospitalType" 
            value={patientData.hospitalType} 
            onChange={handleInputChange} 
          />
        </div>
        
        {/* Add all other input fields similarly */}
        {/* For brevity, I've only included one example above */}
        
        <button onClick={makePrediction}>Predict Hospital Stay</button>
      </div>
      
      {predictionMade && (
        <div className="prediction-section">
          <h2>Prediction Results</h2>
          <p>Predicted length of stay: {prediction}</p>
        </div>
      )}
      
      <div className="stats-section">
        <h2>Hospital Stay Distribution</h2>
        {stayDistribution.length > 0 ? (
          <div className="distribution-chart">
            {/* Add your chart visualization here */}
            <p>Distribution data loaded successfully</p>
          </div>
        ) : (
          <p>Loading stay distribution data...</p>
        )}
      </div>
    </div>
  );
}

export default HospitalDashboard;