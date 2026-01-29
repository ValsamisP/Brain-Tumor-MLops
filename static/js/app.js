// API Base URL
const API_URL = 'http://localhost:8000';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const clearBtn = document.getElementById('clearBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const analyzeBtnText = document.getElementById('analyzeBtnText');
const spinner = document.getElementById('spinner');
const resultsSection = document.getElementById('resultsSection');
const predictionLabel = document.getElementById('predictionLabel');
const confidenceValue = document.getElementById('confidenceValue');
const predictionTime = document.getElementById('predictionTime');
const probabilityBars = document.getElementById('probabilityBars');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const downloadReportBtn = document.getElementById('downloadReportBtn');
const historyList = document.getElementById('historyList');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');

// State
let currentFile = null;
let currentPrediction = null;
let predictionHistory = [];

// Class display names
const CLASS_NAMES = {
    'glioma': 'Glioma Tumor',
    'glioma_tumor': 'Glioma Tumor',
    'meningioma': 'Meningioma Tumor',
    'meningioma_tumor': 'Meningioma Tumor',
    'no_tumor': 'No Tumor',
    'pituitary': 'Pituitary Tumor',
    'pituitary_tumor': 'Pituitary Tumor'
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkAPIStatus();
    loadHistory();
    setupEventListeners();
});

// Setup Event Listeners
function setupEventListeners() {
    // Upload events
    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Actions
    clearBtn.addEventListener('click', clearImage);
    analyzeBtn.addEventListener('click', analyzeImage);
    newAnalysisBtn.addEventListener('click', resetForNewAnalysis);
    downloadReportBtn.addEventListener('click', downloadReport);
    clearHistoryBtn.addEventListener('click', clearHistory);
}

// Check API Status
async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy' && data.model_loaded) {
            statusText.textContent = 'Online';
            statusIndicator.style.background = '#f0fdf4';
        } else {
            statusText.textContent = 'Model Loading...';
            statusIndicator.style.background = '#fef3c7';
        }
    } catch (error) {
        statusText.textContent = 'Offline';
        statusIndicator.style.background = '#fee2e2';
        statusIndicator.querySelector('.status-dot').style.background = '#ef4444';
    }
}

// File handling
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        displayImage(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        displayImage(file);
    }
}

function displayImage(file) {
    currentFile = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        document.querySelector('.upload-area').style.display = 'none';
        previewSection.style.display = 'block';
        resultsSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function clearImage() {
    currentFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    document.querySelector('.upload-area').style.display = 'block';
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
}

// Analyze Image
async function analyzeImage() {
    if (!currentFile) return;
    
    // Show loading state
    analyzeBtn.disabled = true;
    analyzeBtnText.textContent = 'Analyzing...';
    spinner.style.display = 'block';
    
    try {
        const formData = new FormData();
        formData.append('file', currentFile);
        
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const data = await response.json();
        currentPrediction = data;
        
        displayResults(data);
        addToHistory(data);
        
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtnText.textContent = 'Analyze Image';
        spinner.style.display = 'none';
    }
}

// Display Results
function displayResults(data) {
    const className = data.predicted_class;
    const displayName = CLASS_NAMES[className] || className;
    const confidence = (data.confidence * 100).toFixed(1);
    
    predictionLabel.textContent = displayName;
    confidenceValue.textContent = confidence + '%';
    predictionTime.textContent = `Processed in ${data.processing_time_ms.toFixed(0)}ms`;
    
    // Display probability bars
    probabilityBars.innerHTML = '';
    Object.entries(data.probabilities).forEach(([cls, prob]) => {
        const displayCls = CLASS_NAMES[cls] || cls;
        const percentage = (prob * 100).toFixed(1);
        
        const barHTML = `
            <div class="probability-bar">
                <div class="probability-label">
                    <span>${displayCls}</span>
                    <span>${percentage}%</span>
                </div>
                <div class="probability-fill-bg">
                    <div class="probability-fill" style="width: ${percentage}%"></div>
                </div>
            </div>
        `;
        probabilityBars.innerHTML += barHTML;
    });
    
    // Show results
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// History Management
function addToHistory(data) {
    const historyItem = {
        ...data,
        thumbnail: imagePreview.src,
        timestamp: new Date().toISOString()
    };
    
    predictionHistory.unshift(historyItem);
    if (predictionHistory.length > 10) {
        predictionHistory.pop();
    }
    
    saveHistory();
    renderHistory();
}

function renderHistory() {
    if (predictionHistory.length === 0) {
        historyList.innerHTML = '<p class="empty-history">No predictions yet. Upload an image to get started!</p>';
        return;
    }
    
    historyList.innerHTML = predictionHistory.map((item, index) => {
        const className = item.predicted_class;
        const displayName = CLASS_NAMES[className] || className;
        const confidence = (item.confidence * 100).toFixed(1);
        const time = new Date(item.timestamp).toLocaleString();
        
        return `
            <div class="history-item">
                <img src="${item.thumbnail}" class="history-thumbnail" alt="Thumbnail">
                <div class="history-info">
                    <div class="history-label">${displayName}</div>
                    <div class="history-confidence">Confidence: ${confidence}%</div>
                    <div class="history-time">${time}</div>
                </div>
            </div>
        `;
    }).join('');
}

function saveHistory() {
    try {
        localStorage.setItem('predictionHistory', JSON.stringify(predictionHistory));
    } catch (e) {
        console.error('Failed to save history:', e);
    }
}

function loadHistory() {
    try {
        const saved = localStorage.getItem('predictionHistory');
        if (saved) {
            predictionHistory = JSON.parse(saved);
            renderHistory();
        }
    } catch (e) {
        console.error('Failed to load history:', e);
    }
}

function clearHistory() {
    if (confirm('Clear all prediction history?')) {
        predictionHistory = [];
        localStorage.removeItem('predictionHistory');
        renderHistory();
    }
}

// Actions
function resetForNewAnalysis() {
    clearImage();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function downloadReport() {
    if (!currentPrediction) return;
    
    const className = currentPrediction.predicted_class;
    const displayName = CLASS_NAMES[className] || className;
    const confidence = (currentPrediction.confidence * 100).toFixed(1);
    
    const report = `
Brain Tumor Classification Report
================================

Prediction: ${displayName}
Confidence: ${confidence}%
Processing Time: ${currentPrediction.processing_time_ms.toFixed(2)}ms
Timestamp: ${new Date(currentPrediction.timestamp).toLocaleString()}
Prediction ID: ${currentPrediction.prediction_id}

Probabilities:
${Object.entries(currentPrediction.probabilities).map(([cls, prob]) => {
    const displayCls = CLASS_NAMES[cls] || cls;
    return `  ${displayCls}: ${(prob * 100).toFixed(2)}%`;
}).join('\n')}

DISCLAIMER: This is for research purposes only. Not for clinical diagnosis.
================================
    `.trim();
    
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `brain-tumor-report-${currentPrediction.prediction_id}.txt`;
    a.click();
    URL.revokeObjectURL(url);
}

// Refresh API status periodically
setInterval(checkAPIStatus, 30000);