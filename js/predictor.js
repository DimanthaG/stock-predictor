// Global state
let currentStockData = null;
let worker = null;

// Initialize Web Worker
function initializeWorker() {
    if (worker) {
        worker.terminate();
    }
    
    worker = new Worker('js/trainWorker.js');
    
    worker.onmessage = function(e) {
        const { type, ...data } = e.data;
        
        switch (type) {
            case 'initialized':
                console.log('Worker initialized:', data.success);
                break;
                
            case 'progress':
                const progress = document.getElementById('progress');
                const details = document.getElementById('training-details');
                
                if (progress) {
                    progress.textContent = `${data.percent}%`;
                }
                
                if (details) {
                    details.textContent = `Iteration ${data.iteration}/1000: error = ${data.error.toFixed(4)}`;
                }
                break;
                
            case 'complete':
                console.log('Training complete, prediction:', data.prediction);
                updateUI(data.prediction);
                updateChart(data.prediction);
                cleanup();
                break;
                
            case 'error':
                console.error('Training error:', data.message);
                const errorDisplay = document.getElementById('error-display');
                if (errorDisplay) {
                    errorDisplay.textContent = data.message;
                }
                setTimeout(cleanup, 3000);
                break;
        }
    };
    
    worker.onerror = function(error) {
        console.error('Worker error:', error);
        alert('An error occurred during training. Please try again.');
        cleanup();
    };
    
    // Initialize the worker
    worker.postMessage({ type: 'init' });
}

// Initialize worker when page loads
window.addEventListener('load', initializeWorker);

// Data normalization functions for neural network input/output
function scaleDown(step) {
    return {
        open: step.open / 138,
        high: step.high / 138,
        low: step.low / 138,
        close: step.close / 138
    };
}

function scaleUp(step) {
    return {
        open: step.open * 138,
        high: step.high * 138,
        low: step.low * 138,
        close: step.close * 138
    };
}

// Prepare training data with non-overlapping sequences
function prepareTrainingData(data) {
    console.log('Preparing training data from', data.length, 'entries');
    const scaledData = data.map(scaleDown);
    const trainingData = [];
    
    // Create non-overlapping sequences of 5 days each
    for (let i = 0; i < scaledData.length - 5; i += 5) {
        const sequence = scaledData.slice(i, i + 5);
        if (sequence.length === 5) {
            trainingData.push(sequence);
        }
    }
    
    console.log('Generated', trainingData.length, 'training sequences');
    return trainingData;
}

// CSV file parser with validation and error handling
function parseCSV(csv) {
    const cleanCsv = csv.replace(/^\uFEFF/, '').trim();
    const lines = cleanCsv.split(/\r?\n/);
    
    if (lines.length < 2) {
        throw new Error('CSV file must contain headers and at least one data row');
    }
    
    const headers = lines[0].toLowerCase().split(',').map(h => h.trim());
    const requiredColumns = ['date', 'open', 'high', 'low', 'close'];
    
    for (const col of requiredColumns) {
        if (!headers.includes(col)) {
            throw new Error(`CSV must have ${requiredColumns.join(', ')} columns`);
        }
    }
    
    const dateIndex = headers.indexOf('date');
    const openIndex = headers.indexOf('open');
    const highIndex = headers.indexOf('high');
    const lowIndex = headers.indexOf('low');
    const closeIndex = headers.indexOf('close');
    
    const data = [];
    let errorLines = [];
    
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        
        const values = line.split(',').map(v => v.trim());
        
        if (values.length !== headers.length) {
            errorLines.push(`Line ${i + 1}: Expected ${headers.length} columns, found ${values.length}`);
            continue;
        }
        
        const dateStr = values[dateIndex];
        const date = new Date(dateStr);
        if (isNaN(date.getTime())) {
            errorLines.push(`Line ${i + 1}: Invalid date format "${dateStr}"`);
            continue;
        }
        
        const ohlc = {
            open: parseFloat(values[openIndex].replace('$', '')),
            high: parseFloat(values[highIndex].replace('$', '')),
            low: parseFloat(values[lowIndex].replace('$', '')),
            close: parseFloat(values[closeIndex].replace('$', ''))
        };
        
        if (Object.values(ohlc).some(val => isNaN(val) || val <= 0)) {
            errorLines.push(`Line ${i + 1}: Invalid OHLC values`);
            continue;
        }
        
        if (ohlc.low > ohlc.high || ohlc.open > ohlc.high || ohlc.close > ohlc.high || 
            ohlc.low > ohlc.open || ohlc.low > ohlc.close) {
            errorLines.push(`Line ${i + 1}: Invalid OHLC relationships`);
            continue;
        }
        
        data.push({
            date: date.toISOString().split('T')[0],
            ...ohlc
        });
    }
    
    if (data.length < 50) {
        throw new Error(`CSV must contain at least 50 valid data points. Found ${data.length} valid points.${
            errorLines.length ? '\n\nErrors found:\n' + errorLines.join('\n') : ''
        }`);
    }
    
    if (errorLines.length > 0) {
        console.warn('Warning: Some lines were skipped due to errors:\n' + errorLines.join('\n'));
    }
    
    data.sort((a, b) => new Date(a.date) - new Date(b.date));
    return { data };
}

// Add function to show/hide chart loading
function showChartLoading(chartId, message = 'Generating chart...') {
    const container = document.getElementById(chartId).closest('.chart-container');
    let loading = container.querySelector('.chart-loading');
    
    if (!loading) {
        loading = document.createElement('div');
        loading.className = 'chart-loading';
        loading.innerHTML = `
            <div class="chart-loading-content">
                <div class="chart-loading-spinner"></div>
                <div class="chart-loading-text">${message}</div>
            </div>
        `;
        container.appendChild(loading);
    }
}

function hideChartLoading(chartId) {
    const container = document.getElementById(chartId).closest('.chart-container');
    const loading = container.querySelector('.chart-loading');
    if (loading) {
        loading.remove();
    }
}

// Also show loading when handling file upload
async function handleFileUpload(file) {
    try {
        showChartLoading('historicalChart', 'Processing data...');
        
        const text = await file.text();
        let result;
        
        if (file.name.endsWith('.csv')) {
            result = parseCSV(text);
        } else if (file.name.endsWith('.json')) {
            const jsonData = JSON.parse(text);
            result = { data: jsonData };
        } else {
            throw new Error('Unsupported file format');
        }
        
        currentStockData = result.data;
        
        await updateChart();
        document.getElementById('trainButton').disabled = false;
        
    } catch (error) {
        alert('Error processing file: ' + error.message);
        currentStockData = null;
        document.getElementById('trainButton').disabled = true;
        clearCharts();
    } finally {
        hideChartLoading('historicalChart');
    }
}

// Main training function
async function trainAndPredict() {
    console.log('Training started...');
    
    if (!currentStockData) {
        alert('Please upload data first');
        return;
    }
    
    console.log('Current stock data:', currentStockData.length, 'entries');
    
    const existingOverlay = document.querySelector('.loading-overlay');
    if (existingOverlay) {
        document.body.removeChild(existingOverlay);
    }
    
    const loadingOverlay = document.createElement('div');
    loadingOverlay.className = 'loading-overlay';
    loadingOverlay.innerHTML = `
        <div class="loading-content">
            <div class="loading-spinner"></div>
            <div class="loading-text">Training model... <span id="progress">0%</span></div>
            <div class="loading-details" id="training-details" style="margin-top: 10px; font-size: 0.9em; color: #666;"></div>
            <div class="loading-error" id="error-display" style="color: red; margin-top: 10px;"></div>
        </div>
    `;
    document.body.appendChild(loadingOverlay);
    
    const trainButton = document.getElementById('trainButton');
    trainButton.disabled = true;
    
    try {
        // Start training in worker
        worker.postMessage({
            type: 'train',
            data: currentStockData
        });
    } catch (error) {
        console.error('Error starting training:', error);
        cleanup();
        alert('Failed to start training. Please try again.');
    }
}

// Cleanup function
function cleanup() {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay && overlay.parentNode) {
        overlay.parentNode.removeChild(overlay);
    }
    
    const trainButton = document.getElementById('trainButton');
    if (trainButton) {
        trainButton.disabled = false;
    }
}

// Chart management functions
function clearCharts() {
    if (window.historicalChart instanceof Chart) {
        window.historicalChart.destroy();
    }
    if (window.predictionChart instanceof Chart) {
        window.predictionChart.destroy();
    }
    
    const historicalCtx = document.getElementById('historicalChart').getContext('2d');
    const predictionCtx = document.getElementById('predictionChart').getContext('2d');
    historicalCtx.clearRect(0, 0, historicalCtx.canvas.width, historicalCtx.canvas.height);
    predictionCtx.clearRect(0, 0, predictionCtx.canvas.width, predictionCtx.canvas.height);
}

// UI update functions for displaying predictions
function updateUI(prediction) {
    const lastData = currentStockData[currentStockData.length - 1];
    const trend = prediction.close > lastData.close ? 'Up ↑' : 'Down ↓';
    
    document.getElementById('lastOpen').textContent = `$${lastData.open.toFixed(2)}`;
    document.getElementById('lastHigh').textContent = `$${lastData.high.toFixed(2)}`;
    document.getElementById('lastLow').textContent = `$${lastData.low.toFixed(2)}`;
    document.getElementById('lastClose').textContent = `$${lastData.close.toFixed(2)}`;
    
    document.getElementById('predictedOpen').textContent = `$${prediction.open.toFixed(2)}`;
    document.getElementById('predictedHigh').textContent = `$${prediction.high.toFixed(2)}`;
    document.getElementById('predictedLow').textContent = `$${prediction.low.toFixed(2)}`;
    document.getElementById('predictedClose').textContent = `$${prediction.close.toFixed(2)}`;
    
    document.getElementById('predictedTrend').textContent = trend;
    document.getElementById('predictedTrend').style.color = trend === 'Up ↑' ? 'green' : 'red';
}

// Update the chart rendering function
async function updateChart(prediction = null) {
    if (!currentStockData) return;
    
    // Show loading for both charts
    showChartLoading('historicalChart', 'Generating historical chart...');
    if (prediction) {
        showChartLoading('predictionChart', 'Generating prediction chart...');
    }
    
    // Clear existing charts
    clearCharts();
    
    try {
        // Add small delay to ensure loading is visible
        await new Promise(resolve => setTimeout(resolve, 100));
        
        const historicalCtx = document.getElementById('historicalChart').getContext('2d');
        const labels = currentStockData.map(item => item.date);
        const prices = currentStockData.map(item => item.close);
        
        window.historicalChart = new Chart(historicalCtx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Historical Price',
                    data: prices,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'Price ($)'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Historical Stock Prices'
                    }
                }
            }
        });
        
        // Hide historical chart loading
        hideChartLoading('historicalChart');
        
        if (prediction !== null) {
            const predictionCtx = document.getElementById('predictionChart').getContext('2d');
            
            const last10Days = labels.slice(-10);
            const last10Prices = prices.slice(-10);
            
            const nextDate = new Date(labels[labels.length - 1]);
            nextDate.setDate(nextDate.getDate() + 1);
            const predictionDate = nextDate.toISOString().split('T')[0];
            
            const predictionLine = new Array(10).fill(null);
            predictionLine[9] = last10Prices[9];
            predictionLine.push(prediction.close);
            
            window.predictionChart = new Chart(predictionCtx, {
                type: 'line',
                data: {
                    labels: [...last10Days, predictionDate],
                    datasets: [
                        {
                            label: 'Historical Price',
                            data: last10Prices,
                            borderColor: 'rgb(75, 192, 192)',
                            tension: 0.1,
                            fill: false,
                            pointRadius: 4
                        },
                        {
                            label: 'Prediction',
                            data: predictionLine,
                            borderColor: 'rgb(255, 99, 132)',
                            borderDash: [5, 5],
                            tension: 0.1,
                            fill: false,
                            pointRadius: 6,
                            pointStyle: 'circle'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: false,
                            title: {
                                display: true,
                                text: 'Price ($)'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Price Prediction (Last 10 Days + Prediction)'
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const value = context.raw;
                                    if (value === null) return '';
                                    return `$${value.toFixed(2)}`;
                                }
                            }
                        }
                    }
                }
            });
            
            // Hide prediction chart loading
            hideChartLoading('predictionChart');
        }
    } catch (error) {
        console.error('Error generating charts:', error);
        hideChartLoading('historicalChart');
        hideChartLoading('predictionChart');
    }
}

// Event listeners for user interactions
document.getElementById('dataFile').addEventListener('change', function(e) {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

document.getElementById('trainButton').addEventListener('click', trainAndPredict);
