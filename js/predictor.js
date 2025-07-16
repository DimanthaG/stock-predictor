// Global state and neural network initialization
let currentStockData = null;
let net = null;

// Initialize brain.js network with error handling and retry
function initializeNetwork() {
    if (net !== null) return; // Already initialized
    
    if (typeof brain === 'undefined') {
        console.error('Brain.js is not loaded yet. Retrying in 500ms...');
        setTimeout(initializeNetwork, 500);
        return;
    }
    
    try {
        console.log('Initializing neural network...');
        net = new brain.recurrent.LSTMTimeStep({
            inputSize: 4,
            hiddenLayers: [8, 8],
            outputSize: 4
        });
        console.log('Neural network initialized successfully');
    } catch (error) {
        console.error('Failed to initialize neural network:', error);
        setTimeout(initializeNetwork, 500);
    }
}

// Try to initialize immediately
initializeNetwork();

// Also try on window load in case the script loaded before brain.js
window.addEventListener('load', initializeNetwork);

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

// File upload handler for CSV and JSON formats
async function handleFileUpload(file) {
    try {
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
        
        updateChart();
        document.getElementById('trainButton').disabled = false;
        
    } catch (error) {
        alert('Error processing file: ' + error.message);
        currentStockData = null;
        document.getElementById('trainButton').disabled = true;
        clearCharts();
    }
}

// Main training and prediction function
async function trainAndPredict() {
    console.log('Training started...');
    
    if (!net) {
        const error = 'Neural network not initialized. Brain.js may not have loaded correctly.';
        console.error(error);
        alert(error);
        return;
    }
    
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
    
    const cleanup = () => {
        const overlay = document.querySelector('.loading-overlay');
        if (overlay && overlay.parentNode) {
            overlay.parentNode.removeChild(overlay);
        }
        if (trainButton) {
            trainButton.disabled = false;
        }
    };
    
    try {
        const trainingData = prepareTrainingData(currentStockData);
        
        if (trainingData.length === 0) {
            throw new Error('No valid training sequences generated');
        }

        let currentIteration = 0;
        
        await new Promise((resolve, reject) => {
            try {
                net.train(trainingData, {
                    learningRate: 0.005,
                    errorThresh: 0.02,
                    iterations: 1000,
                    log: (stats) => {
                        currentIteration++;
                        const error = typeof stats === 'object' ? stats.error : stats;
                        
                        const progress = document.getElementById('progress');
                        const details = document.getElementById('training-details');
                        
                        if (progress) {
                            const percent = Math.round((currentIteration / 1000) * 100);
                            progress.textContent = `${percent}%`;
                        }
                        
                        if (details && typeof error === 'number') {
                            details.textContent = `Iteration ${currentIteration}/1000: error = ${error.toFixed(4)}`;
                        }
                    },
                    logPeriod: 10
                });
                resolve();
            } catch (error) {
                reject(error);
            }
        });
        
        // Get the last sequence for prediction
        const lastSequence = trainingData[trainingData.length - 1];
        console.log('Last sequence for prediction:', lastSequence);
        
        // Run prediction
        const prediction = scaleUp(net.run(lastSequence));
        console.log('Raw prediction:', prediction);
        
        updateUI(prediction);
        updateChart(prediction);
        
    } catch (error) {
        console.error('Training error:', error);
        const errorDisplay = document.getElementById('error-display');
        if (errorDisplay) {
            errorDisplay.textContent = error.message;
        }
        setTimeout(cleanup, 3000);
        return;
    }
    
    cleanup();
}

// Chart management functions
function clearCharts() {
    const charts = ['historicalChart', 'predictionChart'];
    
    for (const chartId of charts) {
        const canvas = document.getElementById(chartId);
        if (!canvas) continue;
        
        const ctx = canvas.getContext('2d');
        
        // Clear any existing chart
        if (window[chartId] instanceof Chart) {
            window[chartId].destroy();
        }
        
        // Clear the canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

// UI update functions for displaying predictions
function updateUI(prediction) {
    const lastData = currentStockData[currentStockData.length - 1];
    const trend = prediction.close > lastData.close ? 'Up ↑' : 'Down ↓';
    
    // Format currency values
    const formatPrice = (price) => `$${price.toFixed(2)}`;
    
    // Update last known values
    document.getElementById('lastOpen').textContent = formatPrice(lastData.open);
    document.getElementById('lastHigh').textContent = formatPrice(lastData.high);
    document.getElementById('lastLow').textContent = formatPrice(lastData.low);
    document.getElementById('lastClose').textContent = formatPrice(lastData.close);
    
    // Update predicted values
    document.getElementById('predictedOpen').textContent = formatPrice(prediction.open);
    document.getElementById('predictedHigh').textContent = formatPrice(prediction.high);
    document.getElementById('predictedLow').textContent = formatPrice(prediction.low);
    document.getElementById('predictedClose').textContent = formatPrice(prediction.close);
    
    // Update trend with color
    const trendElement = document.getElementById('predictedTrend');
    trendElement.textContent = trend;
    trendElement.style.color = trend === 'Up ↑' ? '#22c55e' : '#ef4444';
    
    // Log prediction details
    console.log('Last known values:', lastData);
    console.log('Predicted values:', prediction);
    console.log('Trend:', trend);
}

// Chart rendering for historical data and predictions
function updateChart(prediction = null) {
    if (!currentStockData) return;
    
    clearCharts();
    
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
    
    if (prediction !== null) {
        const predictionCtx = document.getElementById('predictionChart').getContext('2d');
        
        // Get the last 10 actual data points
        const last10Days = labels.slice(-10);
        const last10Prices = prices.slice(-10);
        
        // Calculate the next date for prediction
        const lastDate = new Date(labels[labels.length - 1]);
        const nextDate = new Date(lastDate);
        nextDate.setDate(nextDate.getDate() + 1);
        const predictionDate = nextDate.toISOString().split('T')[0];
        
        // Create prediction line that connects the last actual price to the predicted price
        const predictionLine = Array(10).fill(null);
        predictionLine[9] = last10Prices[last10Prices.length - 1]; // Last actual price
        predictionLine.push(prediction.close); // Predicted price
        
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
                        pointRadius: [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 6] // Only show points for last actual and prediction
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
                        text: 'Price Prediction (Last 10 Days + Next Day)'
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
    }
}

// Event listeners for user interactions
document.getElementById('dataFile').addEventListener('change', function(e) {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

document.getElementById('trainButton').addEventListener('click', trainAndPredict);
