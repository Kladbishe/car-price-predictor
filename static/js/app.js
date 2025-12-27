const API_BASE_URL = 'http://localhost:5001/api';

// DOM Elements
const brandSelect = document.getElementById('brand');
const modelSelect = document.getElementById('model');
const yearSelect = document.getElementById('year');
const carForm = document.getElementById('carForm');
const resultCard = document.getElementById('resultCard');
const predictBtn = document.getElementById('predictBtn');

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    loadBrands();
    loadStats();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    brandSelect.addEventListener('change', handleBrandChange);
    modelSelect.addEventListener('change', handleModelChange);
    carForm.addEventListener('submit', handleFormSubmit);
}

// Load available brands
async function loadBrands() {
    try {
        const response = await fetch(`${API_BASE_URL}/brands`);
        const data = await response.json();

        if (data.brands) {
            brandSelect.innerHTML = '<option value="">בחר יצרן...</option>';
            data.brands.forEach(brand => {
                const option = document.createElement('option');
                option.value = brand;
                option.textContent = brand;
                brandSelect.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Error loading brands:', error);
        showError('שגיאה בטעינת רשימת היצרנים');
    }
}

// Load statistics
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        const data = await response.json();

        if (data.total_cars) {
            document.getElementById('totalCars').textContent = data.total_cars.toLocaleString('he-IL');
            document.getElementById('brandsCount').textContent = data.brands_count;
            document.getElementById('priceRange').textContent =
                `₪${Math.round(data.price_range.min).toLocaleString('he-IL')} - ₪${Math.round(data.price_range.max).toLocaleString('he-IL')}`;
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Handle brand selection change
async function handleBrandChange() {
    const brand = brandSelect.value;

    // Reset dependent fields
    modelSelect.innerHTML = '<option value="">בחר דגם...</option>';
    modelSelect.disabled = true;
    yearSelect.innerHTML = '<option value="">בחר שנה...</option>';
    yearSelect.disabled = true;
    resultCard.style.display = 'none';

    if (!brand) return;

    try {
        const response = await fetch(`${API_BASE_URL}/models/${encodeURIComponent(brand)}`);
        const data = await response.json();

        if (data.models) {
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelSelect.appendChild(option);
            });
            modelSelect.disabled = false;
        }
    } catch (error) {
        console.error('Error loading models:', error);
        showError('שגיאה בטעינת רשימת הדגמים');
    }
}

// Handle model selection change
async function handleModelChange() {
    const brand = brandSelect.value;
    const model = modelSelect.value;

    // Reset year field
    yearSelect.innerHTML = '<option value="">בחר שנה...</option>';
    yearSelect.disabled = true;
    resultCard.style.display = 'none';

    if (!brand || !model) return;

    try {
        const response = await fetch(`${API_BASE_URL}/years/${encodeURIComponent(brand)}/${encodeURIComponent(model)}`);
        const data = await response.json();

        if (data.years) {
            data.years.forEach(year => {
                const option = document.createElement('option');
                option.value = year;
                option.textContent = year;
                yearSelect.appendChild(option);
            });
            yearSelect.disabled = false;
        }
    } catch (error) {
        console.error('Error loading years:', error);
        showError('שגיאה בטעינת רשימת השנים');
    }
}

// Handle form submission
async function handleFormSubmit(e) {
    e.preventDefault();

    const formData = new FormData(carForm);
    const data = {
        brand: formData.get('brand'),
        model: formData.get('model'),
        year: parseInt(formData.get('year')),
        mileage_km: parseInt(formData.get('mileage_km')) || 100000,
        hand: parseInt(formData.get('hand')) || 1,
        engine_capacity: parseInt(formData.get('engine_capacity')) || 1600,
        city: formData.get('city') || 'Unknown'
    };

    // Validate required fields
    if (!data.brand || !data.model || !data.year) {
        showError('נא למלא את כל השדות הנדרשים');
        return;
    }

    // Show loading state
    setLoadingState(true);

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (result.success) {
            displayResult(result);
        } else {
            showError(result.error || 'שגיאה בחישוב המחיר');
        }
    } catch (error) {
        console.error('Error predicting price:', error);
        showError('שגיאה בחיבור לשרת');
    } finally {
        setLoadingState(false);
    }
}

// Display prediction result
function displayResult(result) {
    const { input, predicted_price, formatted_price } = result;

    // Update car title
    document.getElementById('carTitle').textContent =
        `${input.brand} ${input.model} (${input.year})`;

    // Update car details
    const detailsHTML = `
        <div><strong>קילומטראז':</strong> ${input.mileage_km.toLocaleString('he-IL')} ק"מ</div>
        <div><strong>יד:</strong> ${getHandText(input.hand)}</div>
        <div><strong>נפח מנוע:</strong> ${input.engine_capacity} סמ"ק</div>
        ${input.city !== 'Unknown' ? `<div><strong>עיר:</strong> ${input.city}</div>` : ''}
    `;
    document.getElementById('carDetails').innerHTML = detailsHTML;

    // Update predicted price
    document.getElementById('predictedPrice').textContent = formatted_price;

    // Show result card with animation
    resultCard.style.display = 'block';
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Helper function to get hand text in Hebrew
function getHandText(hand) {
    const handTexts = {
        1: 'יד ראשונה',
        2: 'יד שנייה',
        3: 'יד שלישית',
        4: 'יד רביעית',
        5: 'יד חמישית',
        6: 'יד שישית',
        7: 'יד שביעית',
        8: 'יד שמינית ומעלה'
    };
    return handTexts[hand] || `יד ${hand}`;
}

// Show error message
function showError(message) {
    // Remove existing error messages
    const existingError = document.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }

    // Create new error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;

    // Insert before form
    carForm.parentNode.insertBefore(errorDiv, carForm);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

// Set loading state
function setLoadingState(isLoading) {
    const btnText = predictBtn.querySelector('.btn-text');
    const loader = predictBtn.querySelector('.loader');

    if (isLoading) {
        btnText.style.display = 'none';
        loader.style.display = 'inline-block';
        predictBtn.disabled = true;
    } else {
        btnText.style.display = 'inline';
        loader.style.display = 'none';
        predictBtn.disabled = false;
    }
}
