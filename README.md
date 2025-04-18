# Football Match Prediction System

This project scrapes football match data from agones.gr and uses machine learning to predict match outcomes.

## Features

1. Web Scraping
   - Automated data collection from agones.gr
   - Handles dynamic content using Selenium
   - Extracts match results, scores, and team information

2. Data Processing
   - Calculates team performance metrics
   - Generates features for ML model
   - Handles missing data and edge cases

3. Machine Learning
   - Uses XGBoost classifier
   - Includes feature engineering
   - Provides probability estimates for outcomes

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install Chrome browser (required for Selenium)

3. Run the system:
   ```bash
   python main.py
   ```

## Project Structure

- `scraper.py`: Web scraping functionality
- `predictor.py`: ML model and predictions
- `main.py`: Main execution script
- `requirements.txt`: Project dependencies

## Output

The system generates:
1. CSV files with scraped match data
2. Trained ML model files
3. Log files with predictions and performance metrics

## Notes

- The scraper respects website's robots.txt
- Predictions are based on historical performance
- Model accuracy depends on data quality and quantity

## Requirements

- Python 3.8+
- Chrome browser
- Internet connection
