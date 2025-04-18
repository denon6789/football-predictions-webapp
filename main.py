import logging
from datetime import datetime
from scraper import FootballScraper
from predictor import FootballPredictor
import pandas as pd

def setup_logging():
    """Setup logging configuration"""
    logger = logging.getLogger('FootballPredictions')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('football_predictions.log', encoding='utf-8')
    
    # Create formatters and add it to handlers
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    c_handler.setFormatter(logging.Formatter(log_format))
    f_handler.setFormatter(logging.Formatter(log_format))
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

def main():
    logger = setup_logging()
    
    try:
        # Initialize scraper
        scraper = FootballScraper()
        
        # Scrape data
        logger.info("Starting data scraping...")
        df = scraper.scrape_matches()
        
        if df is None or df.empty:
            logger.error("No data was scraped. Exiting.")
            return
            
        # Validate data
        required_columns = ['home_team', 'away_team', 'home_score', 'away_score', 'result']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return
            
        # Validate result values
        valid_results = ['home', 'away', 'draw']
        invalid_results = df[~df['result'].isin(valid_results)]
        if not invalid_results.empty:
            logger.error(f"Found invalid results: {invalid_results['result'].unique()}")
            return
            
        # Save raw data with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_data_file = f"raw_data_{timestamp}.csv"
        df.to_csv(raw_data_file, index=False, encoding='utf-8')
        logger.info(f"Raw data saved to {raw_data_file}")
        
        # Display sample of scraped data
        logger.info("\nSample of scraped data:")
        logger.info("\nFirst few matches:")
        logger.info(df[['home_team', 'away_team', 'home_score', 'away_score', 'result']].head().to_string())
        
        # Basic statistics
        logger.info("\nData Statistics:")
        logger.info(f"Total matches: {len(df)}")
        unique_teams = set(df['home_team'].tolist() + df['away_team'].tolist())
        logger.info(f"Unique teams: {len(unique_teams)}")
        logger.info(f"Results distribution:\n{df['result'].value_counts().to_string()}")
        
        # Initialize predictor
        predictor = FootballPredictor()
        
        # Train model
        try:
            logger.info("\nPreparing features and training model...")
            accuracy = predictor.train_model(df)
            
            if accuracy and accuracy > 0:
                logger.info(f"\nModel trained successfully!")
                logger.info(f"Model accuracy: {accuracy:.2f}")
                
                # Make some example predictions
                logger.info("\nMaking example predictions...")
                teams = list(unique_teams)[:2]  # Get first two teams for example
                if len(teams) >= 2:
                    prediction, probabilities = predictor.predict_match(teams[0], teams[1])
                    if prediction:
                        logger.info(f"\nExample prediction for {teams[0]} vs {teams[1]}:")
                        logger.info(f"Predicted outcome: {prediction}")
                        logger.info("Probabilities:")
                        for outcome, prob in probabilities.items():
                            logger.info(f"{outcome}: {prob:.2f}")
            else:
                logger.error("Model training failed")
        
        except Exception as e:
            logger.error(f"Model training/prediction failed: {str(e)}")
            if hasattr(e, 'args'):
                for arg in e.args:
                    logger.error(f"Error detail: {arg}")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        if hasattr(e, 'args'):
            for arg in e.args:
                logger.error(f"Error detail: {arg}")

if __name__ == "__main__":
    main()
