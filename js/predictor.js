// Global state and neural network initialization
let currentStockData = null;
let net = null;

// Initialize brain.js network with error handling and retry
function initializeNetwork() {
    if (net !== null) return; // Already initialized
    
    if (typeof brain === 'undefined') {
        console.error('Brain.js is not loaded yet. Retrying in 500ms...');
        // Retry in 500ms
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
        // Retry in 500ms in case of error
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
        open: step.open / window.maxPrice,
        high: step.high / window.maxPrice,
        low: step.low / window.maxPrice,
        close: step.close / window.maxPrice
    };
}

function scaleUp(step) {
    return {
        open: step.open * window.maxPrice,
        high: step.high * window.maxPrice,
        low: step.low * window.maxPrice,
        close: step.close * window.maxPrice
    };
}

// Prepare training data with size limit and data sampling
function prepareTrainingData(data) {
    // Limit to last 365 days of data to make training manageable
    const limitedData = data.slice(-365);
    console.log(`Using last ${limitedData.length} days of data for training`);
    
    const scaledData = limitedData.map(scaleDown);
    const trainingData = [];
    
    // Use smaller sequences (3 days instead of 5) and take every other sequence
    for (let i = 0; i < scaledData.length - 3; i += 2) {
        const sequence = scaledData.slice(i, i + 3);
        if (sequence.length === 3) {
            trainingData.push(sequence);
        }
    }
    
    console.log(`Generated ${trainingData.length} training sequences`);
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
    let maxPrice = 0;
    
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        
        const values = line.split(',').map(v => v.trim().replace('$', ''));
        const prices = [
            parseFloat(values[openIndex]),
            parseFloat(values[highIndex]),
            parseFloat(values[lowIndex]),
            parseFloat(values[closeIndex])
        ];
        
        const validPrices = prices.filter(p => !isNaN(p) && p > 0);
        if (validPrices.length === 4) {
            maxPrice = Math.max(maxPrice, ...validPrices);
        }
    }
    
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
    return { data, maxPrice };
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
            const maxPrice = Math.max(...jsonData.flatMap(d => [d.open, d.high, d.low, d.close]));
            result = { data: jsonData, maxPrice };
        } else {
            throw new Error('Unsupported file format');
        }
        
        currentStockData = result.data;
        window.maxPrice = result.maxPrice;
        
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
        const maxIterations = 100; // Reduce iterations for faster training
        
        await new Promise((resolve, reject) => {
            try {
                net.train(trainingData, {
                    learningRate: 0.01, // Increased learning rate
                    errorThresh: 0.05, // Increased error threshold
                    iterations: maxIterations,
                    log: (stats) => {
                        currentIteration++;
                        const error = typeof stats === 'object' ? stats.error : stats;
                        
                        const progress = document.getElementById('progress');
                        const details = document.getElementById('training-details');
                        
                        if (progress) {
                            const percent = Math.round((currentIteration / maxIterations) * 100);
                            progress.textContent = `${percent}%`;
                        }
                        
                        if (details && typeof error === 'number') {
                            details.textContent = `Iteration ${currentIteration}/${maxIterations}: error = ${error.toFixed(4)}`;
                        }
                        
                        // Force UI update
                        if (currentIteration % 5 === 0) {
                            setTimeout(() => {}, 0);
                        }
                    },
                    logPeriod: 1 // Log every iteration
                });
                resolve();
            } catch (error) {
                reject(error);
            }
        });
        
        // Use last 3 days for prediction instead of 5
        const lastSequence = trainingData[trainingData.length - 1];
        const prediction = scaleUp(net.run(lastSequence));
        
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
    }
}

// Event listeners for user interactions
document.getElementById('dataFile').addEventListener('change', function(e) {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

document.getElementById('trainButton').addEventListener('click', trainAndPredict);
