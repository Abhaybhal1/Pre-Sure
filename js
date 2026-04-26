// JavaScript for Pregnancy Risk Predictor

document.addEventListener('DOMContentLoaded', function() {
    // Initialize form validation and interactions
    initializeForm();
    initializeAnimations();
    initializeTooltips();
});

function initializeForm() {
    const form = document.getElementById('pregnancyForm');
    const predictBtn = document.getElementById('predictBtn');
    
    if (form && predictBtn) {
        // Form validation
        form.addEventListener('submit', function(e) {
            if (!validateForm()) {
                e.preventDefault();
                return false;
            }
            
            // Show loading state
            showLoading(predictBtn);
        });
        
        // Real-time validation
        const inputs = form.querySelectorAll('input[required]');
        inputs.forEach(input => {
            input.addEventListener('blur', function() {
                validateField(this);
            });
            
            input.addEventListener('input', function() {
                clearValidation(this);
            });
        });
    }
}

function validateForm() {
    const form = document.getElementById('pregnancyForm');
    const inputs = form.querySelectorAll('input[required]');
    let isValid = true;
    
    inputs.forEach(input => {
        if (!validateField(input)) {
            isValid = false;
        }
    });
    
    // Custom validation rules
    const age = parseFloat(document.getElementById('Age').value);
    const systolicBP = parseFloat(document.getElementById('SystolicBP').value);
    const diastolicBP = parseFloat(document.getElementById('DiastolicBP').value);
    const bloodSugar = parseFloat(document.getElementById('BS').value);
    const bodyTemp = parseFloat(document.getElementById('BodyTemp').value);
    const heartRate = parseFloat(document.getElementById('HeartRate').value);
    
    // Age validation
    if (age < 15 || age > 50) {
        showFieldError('Age', 'Age should be between 15 and 50 years');
        isValid = false;
    }
    
    // Blood pressure validation
    if (systolicBP <= diastolicBP) {
        showFieldError('SystolicBP', 'Systolic BP should be higher than Diastolic BP');
        showFieldError('DiastolicBP', 'Diastolic BP should be lower than Systolic BP');
        isValid = false;
    }
    
    // Blood pressure ranges
    if (systolicBP < 80 || systolicBP > 200) {
        showFieldError('SystolicBP', 'Systolic BP should be between 80-200 mmHg');
        isValid = false;
    }
    
    if (diastolicBP < 50 || diastolicBP > 120) {
        showFieldError('DiastolicBP', 'Diastolic BP should be between 50-120 mmHg');
        isValid = false;
    }
    
    // Blood sugar validation
    if (bloodSugar < 50 || bloodSugar > 300) {
        showFieldError('BS', 'Blood sugar should be between 50-300 mg/dL');
        isValid = false;
    }
    
    // Body temperature validation
    if (bodyTemp < 95 || bodyTemp > 105) {
        showFieldError('BodyTemp', 'Body temperature should be between 95-105°F');
        isValid = false;
    }
    
    // Heart rate validation
    if (heartRate < 50 || heartRate > 150) {
        showFieldError('HeartRate', 'Heart rate should be between 50-150 bpm');
        isValid = false;
    }
    
    return isValid;
}

function validateField(field) {
    const value = field.value.trim();
    const fieldName = field.name;
    
    // Required field validation
    if (!value) {
        showFieldError(fieldName, `${fieldName} is required`);
        return false;
    }
    
    // Numeric validation
    if (field.type === 'number') {
        const numValue = parseFloat(value);
        const min = parseFloat(field.min);
        const max = parseFloat(field.max);
        
        if (isNaN(numValue)) {
            showFieldError(fieldName, `${fieldName} must be a valid number`);
            return false;
        }
        
        if (min && numValue < min) {
            showFieldError(fieldName, `${fieldName} must be at least ${min}`);
            return false;
        }
        
        if (max && numValue > max) {
            showFieldError(fieldName, `${fieldName} must be at most ${max}`);
            return false;
        }
    }
    
    // Show success state
    showFieldSuccess(fieldName);
    return true;
}

function showFieldError(fieldName, message) {
    const field = document.getElementById(fieldName);
    const formGroup = field.closest('.mb-3');
    
    // Remove existing validation classes
    field.classList.remove('is-valid', 'is-invalid');
    field.classList.add('is-invalid');
    
    // Remove existing error message
    const existingError = formGroup.querySelector('.invalid-feedback');
    if (existingError) {
        existingError.remove();
    }
    
    // Add error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'invalid-feedback';
    errorDiv.textContent = message;
    field.parentNode.appendChild(errorDiv);
}

function showFieldSuccess(fieldName) {
    const field = document.getElementById(fieldName);
    const formGroup = field.closest('.mb-3');
    
    // Remove existing validation classes and messages
    field.classList.remove('is-valid', 'is-invalid');
    field.classList.add('is-valid');
    
    const existingError = formGroup.querySelector('.invalid-feedback');
    if (existingError) {
        existingError.remove();
    }
}

function clearValidation(field) {
    field.classList.remove('is-valid', 'is-invalid');
    const formGroup = field.closest('.mb-3');
    const existingError = formGroup.querySelector('.invalid-feedback');
    if (existingError) {
        existingError.remove();
    }
}

function showLoading(button) {
    const originalText = button.innerHTML;
    button.disabled = true;
    button.innerHTML = `
        <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
        Analyzing...
    `;
    
    // Store original text for restoration
    button.setAttribute('data-original-text', originalText);
}

function hideLoading(button) {
    const originalText = button.getAttribute('data-original-text');
    button.disabled = false;
    button.innerHTML = originalText || button.innerHTML;
    button.removeAttribute('data-original-text');
}

function initializeAnimations() {
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in-up');
    });
    
    // Add hover effects to feature boxes
    const featureBoxes = document.querySelectorAll('.feature-box, .card');
    featureBoxes.forEach(box => {
        box.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
        });
        
        box.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
}

function initializeTooltips() {
    // Initialize Bootstrap tooltips if available
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}

// Risk level indicator functions
function showRiskIndicators() {
    const form = document.getElementById('pregnancyForm');
    if (!form) return;
    
    const inputs = form.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            updateRiskIndicator(this);
        });
    });
}

function updateRiskIndicator(input) {
    const value = parseFloat(input.value);
    const fieldName = input.name;
    let riskLevel = 'normal';
    
    // Define risk thresholds
    const riskThresholds = {
        'Age': { high: [15, 19, 35, 50], medium: [20, 34] },
        'SystolicBP': { high: [140, 200], medium: [130, 139], normal: [90, 129] },
        'DiastolicBP': { high: [90, 120], medium: [85, 89], normal: [60, 84] },
        'BS': { high: [126, 300], medium: [100, 125], normal: [70, 99] },
        'BodyTemp': { high: [100.4, 105], medium: [99.5, 100.3], normal: [97, 99.4] },
        'HeartRate': { high: [100, 150], medium: [90, 99], normal: [60, 89] }
    };
    
    const thresholds = riskThresholds[fieldName];
    if (thresholds) {
        if (value >= thresholds.high[0] && value <= thresholds.high[1]) {
            riskLevel = 'high';
        } else if (thresholds.medium && value >= thresholds.medium[0] && value <= thresholds.medium[1]) {
            riskLevel = 'medium';
        } else if (thresholds.normal && value >= thresholds.normal[0] && value <= thresholds.normal[1]) {
            riskLevel = 'normal';
        }
    }
    
    // Update field styling based on risk level
    input.classList.remove('border-success', 'border-warning', 'border-danger');
    switch(riskLevel) {
        case 'high':
            input.classList.add('border-danger');
            break;
        case 'medium':
            input.classList.add('border-warning');
            break;
        case 'normal':
            input.classList.add('border-success');
            break;
    }
}

// API functions for programmatic access
const PregnancyRiskAPI = {
    predict: async function(data) {
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            return result;
        } catch (error) {
            console.error('Error making prediction:', error);
            throw error;
        }
    },
    
    validateInput: function(data) {
        const errors = [];
        
        // Validate required fields
        const requiredFields = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'];
        requiredFields.forEach(field => {
            if (!data[field] || isNaN(data[field])) {
                errors.push(`${field} is required and must be a number`);
            }
        });
        
        // Validate ranges
        if (data.Age < 15 || data.Age > 50) errors.push('Age must be between 15-50 years');
        if (data.SystolicBP < 80 || data.SystolicBP > 200) errors.push('Systolic BP must be between 80-200 mmHg');
        if (data.DiastolicBP < 50 || data.DiastolicBP > 120) errors.push('Diastolic BP must be between 50-120 mmHg');
        if (data.BS < 50 || data.BS > 300) errors.push('Blood Sugar must be between 50-300 mg/dL');
        if (data.BodyTemp < 95 || data.BodyTemp > 105) errors.push('Body Temperature must be between 95-105°F');
        if (data.HeartRate < 50 || data.HeartRate > 150) errors.push('Heart Rate must be between 50-150 bpm');
        
        // Validate BP relationship
        if (data.SystolicBP <= data.DiastolicBP) {
            errors.push('Systolic BP must be higher than Diastolic BP');
        }
        
        return errors;
    }
};

// Export API for external use
window.PregnancyRiskAPI = PregnancyRiskAPI;

// Utility functions
function formatNumber(num, decimals = 1) {
    return Number(num).toFixed(decimals);
}

function getHealthStatus(value, thresholds) {
    if (value >= thresholds.high[0] && value <= thresholds.high[1]) return 'high';
    if (value >= thresholds.medium[0] && value <= thresholds.medium[1]) return 'medium';
    return 'normal';
}

// Print functionality
function printResults() {
    window.print();
}

// Copy results functionality
function copyResults() {
    const resultsText = document.querySelector('.card-body').innerText;
    navigator.clipboard.writeText(resultsText).then(function() {
        showNotification('Results copied to clipboard!', 'success');
    }).catch(function() {
        showNotification('Failed to copy results', 'error');
    });
}

// Show notification
function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.parentNode.removeChild(alertDiv);
        }
    }, 3000);
}

// Initialize risk indicators when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    showRiskIndicators();
});
