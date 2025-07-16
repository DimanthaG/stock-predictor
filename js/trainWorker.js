// Import brain.js in the worker
importScripts('https://unpkg.com/brain.js');

let net = null;

// Initialize neural network
function initializeNetwork() {
    try {
        net = new brain.recurrent.LSTMTimeStep({
            inputSize: 4,
            hiddenLayers: [8, 8],
            outputSize: 4
        });
        return true;
    } catch (error) {
        return false;
    }
}

// Data normalization functions
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

// Prepare training data
function prepareTrainingData(data) {
    const scaledData = data.map(scaleDown);
    const trainingData = [];
    
    for (let i = 0; i < scaledData.length - 5; i += 5) {
        const sequence = scaledData.slice(i, i + 5);
        if (sequence.length === 5) {
            trainingData.push(sequence);
        }
    }
    
    return trainingData;
}

// Handle messages from main thread
self.onmessage = function(e) {
    const { type, data } = e.data;
    
    switch (type) {
        case 'init':
            const initialized = initializeNetwork();
            self.postMessage({ type: 'initialized', success: initialized });
            break;
            
        case 'train':
            try {
                if (!net) {
                    throw new Error('Neural network not initialized');
                }
                
                const trainingData = prepareTrainingData(data);
                let currentIteration = 0;
                
                net.train(trainingData, {
                    learningRate: 0.005,
                    errorThresh: 0.02,
                    iterations: 1000,
                    log: (stats) => {
                        currentIteration++;
                        const error = typeof stats === 'object' ? stats.error : stats;
                        
                        // Report progress every 10 iterations
                        if (currentIteration % 10 === 0) {
                            self.postMessage({
                                type: 'progress',
                                iteration: currentIteration,
                                error: error,
                                percent: Math.round((currentIteration / 1000) * 100)
                            });
                        }
                    },
                    logPeriod: 10
                });
                
                // Get prediction
                const lastSequence = trainingData[trainingData.length - 1];
                const prediction = scaleUp(net.run(lastSequence));
                
                self.postMessage({
                    type: 'complete',
                    prediction: prediction
                });
                
            } catch (error) {
                self.postMessage({
                    type: 'error',
                    message: error.message
                });
            }
            break;
    }
}; 