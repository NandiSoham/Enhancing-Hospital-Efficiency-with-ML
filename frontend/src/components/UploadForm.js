import React, { useState } from 'react';
import { uploadFile } from '../api/api';

const UploadForm = ({ setPredictions }) => {
    const [file, setFile] = useState(null);

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await uploadFile(formData);
            setPredictions(response.data.predictions);
        } catch (error) {
            console.error('Error uploading file:', error);
        }
    };

    return (
        <div className="p-4 bg-white rounded-2xl shadow-md">
            <h2 className="text-xl mb-4">Upload Patient Data (CSV)</h2>
            <form onSubmit={handleSubmit}>
                <input type="file" onChange={handleFileChange} className="mb-4" />
                <button type="submit" className="px-4 py-2 bg-blue-500 text-white rounded-xl">
                    Predict
                </button>
            </form>
        </div>
    );
};

export default UploadForm;
