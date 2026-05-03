document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('scan-form');
    const urlInput = document.getElementById('url-input');
    const scanBtn = document.getElementById('scan-btn');
    const btnText = document.querySelector('.btn-text');
    const spinner = document.querySelector('.spinner');
    
    const resultsPanel = document.getElementById('results-panel');
    const probCircle = document.getElementById('prob-circle');
    const probText = document.getElementById('prob-text');
    const verdictStatus = document.getElementById('verdict-status');
    const latencyVal = document.getElementById('latency-val');
    
    let shapChartInstance = null;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const url = urlInput.value.trim();
        if (!url) return;

        // UI Loading State
        btnText.classList.add('hidden');
        spinner.classList.remove('hidden');
        scanBtn.disabled = true;
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: url })
            });
            
            if (!response.ok) throw new Error('API Error');
            const data = await response.json();
            
            updateDashboard(data);
        } catch (error) {
            console.error('Prediction failed:', error);
            alert('Failed to analyze URL. Ensure the API is running and models are loaded.');
        } finally {
            // UI Reset State
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
            scanBtn.disabled = false;
        }
    });

    function updateDashboard(data) {
        // Show results
        resultsPanel.classList.remove('hidden');
        
        // 1. Update Probability Ring
        const probPct = Math.round(data.phishing_probability * 100);
        probText.textContent = `${probPct}%`;
        probCircle.style.strokeDasharray = `${probPct}, 100`;
        
        // Colors: Safe (Green), Phishing (Red)
        const isPhishing = data.is_phishing;
        const color = isPhishing ? 'var(--danger)' : 'var(--safe)';
        probCircle.style.stroke = color;
        
        verdictStatus.textContent = isPhishing ? 'Phishing Detected' : 'Legitimate Site';
        verdictStatus.style.color = color;
        
        // 2. Update Latency
        latencyVal.textContent = data.latency_ms;

        // 3. Render SHAP Chart
        renderShapChart(data.explanation.feature_contributions);
    }

    function renderShapChart(contributions) {
        const ctx = document.getElementById('shapChart').getContext('2d');
        
        // Sort features by absolute impact
        const entries = Object.entries(contributions)
            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
            .slice(0, 10); // Top 10 features for UI fit
            
        const labels = entries.map(e => e[0]);
        const values = entries.map(e => e[1]);
        const colors = values.map(v => v > 0 ? 'rgba(239, 68, 68, 0.8)' : 'rgba(16, 185, 129, 0.8)');
        const borderColors = values.map(v => v > 0 ? 'rgb(239, 68, 68)' : 'rgb(16, 185, 129)');

        if (shapChartInstance) {
            shapChartInstance.destroy();
        }

        shapChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'SHAP Value (Impact on Output)',
                    data: values,
                    backgroundColor: colors,
                    borderColor: borderColors,
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let val = context.raw.toFixed(3);
                                let meaning = val > 0 ? "Pushes towards Phishing" : "Pushes towards Legitimate";
                                return `Impact: ${val} (${meaning})`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.7)' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: 'rgba(255, 255, 255, 0.9)' }
                    }
                }
            }
        });
    }
});
