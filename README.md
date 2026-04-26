# Pre-Sure 🤱

An AI-powered web application for assessing pregnancy risk levels based on medical parameters. This tool uses machine learning algorithms to analyze vital signs and provide risk assessments with actionable recommendations.

## 🚨 Disclaimer

**This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.**

## ✨ Features

- **AI-Powered Risk Assessment**: Uses multiple machine learning algorithms (Random Forest, Gradient Boosting, Logistic Regression, SVM)
- **Real-time Predictions**: Instant risk level assessment with confidence scores
- **Interactive Web Interface**: User-friendly form with real-time validation
- **Comprehensive Analysis**: Considers age, blood pressure, blood sugar, body temperature, and heart rate
- **Risk Categories**: Classifies into Low, Medium, and High risk levels
- **Medical Recommendations**: Provides actionable advice based on risk level
- **Model Performance Metrics**: Detailed accuracy and performance statistics
- **Feature Importance Analysis**: Shows which factors contribute most to predictions
- **Mobile-Responsive Design**: Works seamlessly across all devices
- **API Endpoints**: RESTful API for programmatic access

## 🏗️ Project Structure

```
pregnancy-risk-predictor/
├── app.py                 # Main Flask application
├── model.py               # Machine learning model
├── data/
│   ├── pregnancy_data.csv # Sample dataset
│   └── preprocess.py      # Data preprocessing
├── templates/
│   ├── index.html         # Main page
│   └── results.html       # Results page
├── static/
│   ├── css/
│   │   └── style.css      # Styling
│   └── js/
│       └── script.js      # JavaScript
├── models/
│   └── trained_model.pkl  # Saved model
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .gitignore             # Ignore unnecessary files
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional)

### Installation

1. **Clone the repository** (or download the files):
   ```bash
   git clone https://github.com/your-username/pregnancy-risk-predictor.git
   cd pregnancy-risk-predictor
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv pregnancy_env
   
   # On Windows
   pregnancy_env\Scripts\activate
   
   # On macOS/Linux
   source pregnancy_env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Access the application**:
   Open your browser and go to `http://localhost:5000`

## 📊 Dataset Information

The application uses a synthetic dataset with the following features:

| Feature | Description | Range | Unit |
|---------|-------------|--------|------|
| Age | Mother's age | 15-50 | years |
| SystolicBP | Systolic blood pressure | 80-200 | mmHg |
| DiastolicBP | Diastolic blood pressure | 50-120 | mmHg |
| BS | Blood sugar level | 50-300 | mg/dL |
| BodyTemp | Body temperature | 95-105 | °F |
| HeartRate | Heart rate | 50-150 | bpm |

**Target Variable**: RiskLevel (low risk, mid risk, high risk)

## 🤖 Machine Learning Pipeline

### Data Preprocessing
- **Data Cleaning**: Handle missing values and outliers
- **Feature Encoding**: Convert categorical variables to numerical
- **Feature Scaling**: Standardize numerical features
- **Data Splitting**: 80% training, 20% testing

### Model Training
1. **Multiple Algorithms**: Train and compare 4 different models
2. **Hyperparameter Tuning**: Grid search for optimal parameters
3. **Cross-Validation**: 5-fold CV for robust performance estimation
4. **Feature Importance**: Analyze which features matter most

### Model Selection
- Best model selected based on accuracy and cross-validation scores
- Performance metrics include accuracy, ROC-AUC, and classification reports

## 📈 Model Performance

The application automatically selects the best performing model. Typical performance:

- **Accuracy**: ~85-92%
- **ROC-AUC**: ~0.88-0.95
- **Cross-validation Score**: ~87-90%

## 🔧 Usage

### Web Interface

1. Navigate to the home page
2. Fill in the medical information form:
   - Age (15-50 years)
   - Systolic Blood Pressure (80-200 mmHg)
   - Diastolic Blood Pressure (50-120 mmHg)
   - Blood Sugar (50-300 mg/dL)
   - Body Temperature (95-105°F)
   - Heart Rate (50-150 bpm)
3. Click "Assess Risk Level"
4. View results with risk level, confidence, and recommendations

### API Usage

**Endpoint**: `POST /api/predict`

**Request Body**:
```json
{
    "Age": 25,
    "SystolicBP": 120,
    "DiastolicBP": 80,
    "BS": 95.5,
    "BodyTemp": 98.6,
    "HeartRate": 75
}
```

**Response**:
```json
{
    "success": true,
    "prediction": {
        "risk_level": "Low Risk",
        "confidence": "87.3%",
        "message": "Low risk pregnancy. Continue with regular prenatal care.",
        "recommendations": ["Maintain regular prenatal checkups", "..."]
    },
    "model_name": "RandomForest"
}
```

## 🛠️ Development

### Running in Development Mode

```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
```

### Training Custom Models

```bash
# Run preprocessing
python data/preprocess.py

# Train models
python model.py
```

### Testing

```bash
pytest tests/
```

## 📱 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main application page |
| `/predict` | POST | Web form prediction |
| `/api/predict` | POST | JSON API prediction |
| `/model_info` | GET | Model information and statistics |
| `/health` | GET | Application health check |

## 🎯 Risk Categories

### Low Risk 🟢
- **Indicators**: Normal vital signs within healthy ranges
- **Recommendations**: Regular prenatal care, healthy lifestyle
- **Monitoring**: Standard checkup schedule

### Medium Risk 🟡
- **Indicators**: Some parameters outside normal range
- **Recommendations**: Increased monitoring, dietary modifications
- **Monitoring**: More frequent checkups

### High Risk 🔴
- **Indicators**: Multiple risk factors present
- **Recommendations**: Immediate medical consultation, specialized care
- **Monitoring**: Frequent medical supervision, possible hospitalization

## 🔒 Security and Privacy

- No personal data is stored permanently
- All computations happen locally
- HTTPS recommended for production
- Input validation and sanitization
- No external API calls with sensitive data

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment

#### Using Gunicorn
```bash
pip install gunicorn
gunicorn --bind 0.0.0.0:5000 app:app
```

#### Using Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

#### Environment Variables
```bash
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Ensure medical accuracy

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Medical Disclaimer

**IMPORTANT**: This application is designed for educational and research purposes only. It is not intended to:

- Replace professional medical advice
- Provide medical diagnoses
- Be used for actual medical decision-making
- Substitute consultations with healthcare professionals

Always consult with qualified healthcare providers for:
- Medical advice and treatment
- Pregnancy-related concerns
- Health condition assessments
- Any medical decisions

## 🙏 Acknowledgments

- Medical research data sources for feature selection
- Open-source machine learning libraries
- Bootstrap for responsive UI components
- Font Awesome for icons

## 📞 Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Check the documentation
- Review existing issues and discussions

## 🔮 Future Enhancements

- [ ] Integration with wearable devices
- [ ] Multi-language support
- [ ] Advanced visualization dashboards
- [ ] Historical trend analysis
- [ ] Mobile app development
- [ ] Integration with EHR systems
- [ ] Automated reporting features
- [ ] Real-time monitoring capabilities

## 📊 Technical Specifications

- **Backend**: Python 3.8+, Flask 2.3+
- **ML Libraries**: scikit-learn, pandas, numpy
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **UI Framework**: Bootstrap 5.3
- **Database**: File-based (CSV) or can be extended to SQL
- **API**: RESTful JSON API
- **Performance**: ~100ms prediction time
- **Scalability**: Can handle concurrent requests

---

**Remember: This tool is for educational purposes only. Always consult healthcare professionals for medical advice.** 🏥
