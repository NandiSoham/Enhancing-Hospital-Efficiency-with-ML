import React, { useState } from 'react';
import UploadForm from './components/UploadForm';
import PredictionDashboard from './components/PredictionDashboard';
import ChartVisualization from './components/ChartVisualization';

function App() {
    const [predictions, setPredictions] = useState([]);

    const chartData = predictions.map((pred, index) => ({
        name: `Patient ${index + 1}`,
        value: pred
    }));

    return (
        <div className="min-h-screen bg-gray-100 p-10">
            <h1 className="text-3xl mb-6 text-center">Hospital Stay Prediction</h1>
            <UploadForm setPredictions={setPredictions} />
            {predictions.length > 0 && (
                <>
                    <PredictionDashboard predictions={predictions} />
                    <ChartVisualization data={chartData} />
                </>
            )}
        </div>
    );
}

export default App;
