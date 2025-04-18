import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime

class FootballPredictor:
    def __init__(self):
        self.logger = logging.getLogger('FootballPredictions')
        self.model = None  # Will be initialized during training
        self.team_stats = {}
        self.label_encoder = LabelEncoder()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('predictor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        try:
            # Calculate team statistics
            self.calculate_team_stats(df)
            
            # Create feature matrix
            features = []
            for _, row in df.iterrows():
                home_team = row['home_team']
                away_team = row['away_team']
                
                if home_team not in self.team_stats or away_team not in self.team_stats:
                    continue
                
                home_stats = self.team_stats[home_team]
                away_stats = self.team_stats[away_team]
                
                home_games = max(1, home_stats['games_played'])
                away_games = max(1, away_stats['games_played'])
                
                feature_row = {
                    'home_goals_per_game': home_stats['goals_scored'] / home_games,
                    'home_conceded_per_game': home_stats['goals_conceded'] / home_games,
                    'home_win_rate': home_stats['wins'] / home_games,
                    'home_draw_rate': home_stats['draws'] / home_games,
                    'away_goals_per_game': away_stats['goals_scored'] / away_games,
                    'away_conceded_per_game': away_stats['goals_conceded'] / away_games,
                    'away_win_rate': away_stats['wins'] / away_games,
                    'away_draw_rate': away_stats['draws'] / away_games,
                    'home_form': (home_stats['wins'] * 3 + home_stats['draws']) / (home_games * 3),
                    'away_form': (away_stats['wins'] * 3 + away_stats['draws']) / (away_games * 3)
                }
                features.append(feature_row)
            
            if not features:
                self.logger.error("No valid features could be created")
                return None
            
            return pd.DataFrame(features)
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            if hasattr(e, 'args'):
                for arg in e.args:
                    self.logger.error(f"Error detail: {arg}")
            return None
            
    def calculate_team_stats(self, df):
        """Calculate statistics for each team"""
        try:
            # Initialize team stats
            self.team_stats = {}
            
            # Process each match
            for _, row in df.iterrows():
                home_team = row['home_team']
                away_team = row['away_team']
                home_score = row['home_score']
                away_score = row['away_score']
                
                # Initialize team stats if not exists
                for team in [home_team, away_team]:
                    if team not in self.team_stats:
                        self.team_stats[team] = {
                            'games_played': 0,
                            'goals_scored': 0,
                            'goals_conceded': 0,
                            'wins': 0,
                            'draws': 0,
                            'losses': 0
                        }
                
                # Update home team stats
                self.team_stats[home_team]['games_played'] += 1
                self.team_stats[home_team]['goals_scored'] += home_score
                self.team_stats[home_team]['goals_conceded'] += away_score
                
                if home_score > away_score:
                    self.team_stats[home_team]['wins'] += 1
                elif home_score < away_score:
                    self.team_stats[home_team]['losses'] += 1
                else:
                    self.team_stats[home_team]['draws'] += 1
                
                # Update away team stats
                self.team_stats[away_team]['games_played'] += 1
                self.team_stats[away_team]['goals_scored'] += away_score
                self.team_stats[away_team]['goals_conceded'] += home_score
                
                if away_score > home_score:
                    self.team_stats[away_team]['wins'] += 1
                elif away_score < home_score:
                    self.team_stats[away_team]['losses'] += 1
                else:
                    self.team_stats[away_team]['draws'] += 1
                    
            self.logger.info(f"Calculated statistics for {len(self.team_stats)} teams")
            
        except Exception as e:
            self.logger.error(f"Error calculating team stats: {str(e)}")
            if hasattr(e, 'args'):
                for arg in e.args:
                    self.logger.error(f"Error detail: {arg}")
            self.team_stats = {}
            
    def train_model(self, df):
        """Train the prediction model"""
        try:
            # Prepare features
            features_df = self.prepare_features(df)
            if features_df is None or features_df.empty:
                self.logger.error("No valid features could be prepared")
                return None
            
            # Prepare labels
            labels = df['result'].values
            
            # Check if we have enough data
            min_samples = 3  # Minimum number of samples needed
            if len(features_df) < min_samples:
                self.logger.error(f"Not enough data for training. Need at least {min_samples} samples, got {len(features_df)}")
                return None
            
            # Check class distribution
            unique_classes = len(np.unique(labels))
            if unique_classes < 2:
                self.logger.error(f"Not enough unique classes. Need at least 2, got {unique_classes}")
                return None
            
            # Encode labels
            self.label_encoder.fit(labels)
            encoded_labels = self.label_encoder.transform(labels)
            
            # For very small datasets, use a simple model without train-test split
            if len(features_df) < 5:
                self.model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=len(self.label_encoder.classes_),
                    n_estimators=10,
                    learning_rate=0.3,
                    max_depth=2,
                    random_state=42,
                    min_child_weight=2,
                    use_label_encoder=False
                )
                self.model.fit(features_df, encoded_labels)
                y_pred = self.model.predict(features_df)
                accuracy = accuracy_score(encoded_labels, y_pred)
                self.logger.info("\nTraining Metrics (small dataset):")
                report = classification_report(
                    encoded_labels,
                    y_pred,
                    target_names=self.label_encoder.classes_,
                    zero_division=0
                )
                self.logger.info(f"\n{report}")
            else:
                test_size = 0.2
                X_train, X_test, y_train, y_test = train_test_split(
                    features_df, encoded_labels, test_size=test_size, random_state=42, stratify=encoded_labels
                )
                self.model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=len(self.label_encoder.classes_),
                    n_estimators=50,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42,
                    min_child_weight=2,
                    use_label_encoder=False
                )
                # Only pass early_stopping_rounds if eval_set is used
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=False
                )
                y_pred = self.model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                self.logger.info("\nTest Metrics:")
                report = classification_report(
                    y_test,
                    y_pred,
                    target_names=self.label_encoder.classes_,
                    zero_division=0
                )
                self.logger.info(f"\n{report}")
            joblib.dump(self.model, 'football_predictor.joblib')
            joblib.dump(self.label_encoder, 'label_encoder.joblib')
            return accuracy
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            if hasattr(e, 'args'):
                for arg in e.args:
                    self.logger.error(f"Error detail: {arg}")
            return None
            
    def predict_match(self, home_team, away_team):
        """Predict the outcome of a match"""
        try:
            if self.model is None:
                self.logger.error("Model not trained")
                return None, {}
                
            # Prepare features for prediction
            features = pd.DataFrame([{
                'home_goals_per_game': self.team_stats.get(home_team, {}).get('goals_scored', 0) / max(1, self.team_stats.get(home_team, {}).get('games_played', 1)),
                'home_conceded_per_game': self.team_stats.get(home_team, {}).get('goals_conceded', 0) / max(1, self.team_stats.get(home_team, {}).get('games_played', 1)),
                'home_win_rate': self.team_stats.get(home_team, {}).get('wins', 0) / max(1, self.team_stats.get(home_team, {}).get('games_played', 1)),
                'home_draw_rate': self.team_stats.get(home_team, {}).get('draws', 0) / max(1, self.team_stats.get(home_team, {}).get('games_played', 1)),
                'away_goals_per_game': self.team_stats.get(away_team, {}).get('goals_scored', 0) / max(1, self.team_stats.get(away_team, {}).get('games_played', 1)),
                'away_conceded_per_game': self.team_stats.get(away_team, {}).get('goals_conceded', 0) / max(1, self.team_stats.get(away_team, {}).get('games_played', 1)),
                'away_win_rate': self.team_stats.get(away_team, {}).get('wins', 0) / max(1, self.team_stats.get(away_team, {}).get('games_played', 1)),
                'away_draw_rate': self.team_stats.get(away_team, {}).get('draws', 0) / max(1, self.team_stats.get(away_team, {}).get('games_played', 1)),
                'home_form': (self.team_stats.get(home_team, {}).get('wins', 0) * 3 + self.team_stats.get(home_team, {}).get('draws', 0)) / (max(1, self.team_stats.get(home_team, {}).get('games_played', 1)) * 3),
                'away_form': (self.team_stats.get(away_team, {}).get('wins', 0) * 3 + self.team_stats.get(away_team, {}).get('draws', 0)) / (max(1, self.team_stats.get(away_team, {}).get('games_played', 1)) * 3)
            }])
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(features)[0]
            
            # Get predicted class
            predicted_class = self.label_encoder.inverse_transform([np.argmax(probabilities)])[0]
            
            # Create probabilities dictionary
            prob_dict = {
                class_name: prob 
                for class_name, prob in zip(self.label_encoder.classes_, probabilities)
            }
            
            return predicted_class, prob_dict
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            if hasattr(e, 'args'):
                for arg in e.args:
                    self.logger.error(f"Error detail: {arg}")
            return None, {}
            
    def save_model(self, filename=None):
        """Save the trained model"""
        if filename is None:
            filename = f"football_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        
        try:
            joblib.dump(self.model, filename)
            self.logger.info(f"Model saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
    
    def load_model(self, filename):
        """Load a trained model"""
        try:
            self.model = joblib.load(filename)
            self.logger.info(f"Model loaded from {filename}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")

if __name__ == "__main__":
    # Example usage
    predictor = FootballPredictor()
    predictor.setup_logging()
    
    # Load data
    df = pd.read_csv("matches_latest.csv")  # Replace with your data file
    
    # Prepare features and train model
    accuracy = predictor.train_model(df)
    if accuracy > 0:
        predictor.save_model()
