import React from 'react';

const PredictionDashboard = ({ predictions }) => {
    return (
        <div className="p-4 bg-white rounded-2xl shadow-md mt-6">
            <h2 className="text-xl mb-4">Predictions</h2>
            <ul>
                {predictions.map((prediction, index) => (
                    <li key={index} className="border-b py-2">
                        Patient {index + 1}: Stay Duration - {prediction}
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default PredictionDashboard;
