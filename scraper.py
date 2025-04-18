import logging
import time
import re
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import unicodedata

class FootballScraper:
    def __init__(self):
        self.logger = logging.getLogger('FootballPredictions')
        self.setup_logging()
        
        try:
            # Set up Chrome options
            chrome_options = Options()
            chrome_options.add_argument('--headless=new')  # Use new headless mode
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--lang=el-GR,el')  # Set language to Greek
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Add user agent
            chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            
            # Initialize WebDriver with Service
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Set up wait
            self.wait = WebDriverWait(self.driver, 10)
            
            # Execute CDP commands to prevent detection
            self.driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            # Add missing webdriver properties to prevent detection
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            self.logger.info("Chrome driver setup successful")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Chrome driver: {str(e)}")
            if hasattr(e, 'args'):
                for arg in e.args:
                    self.logger.error(f"Error detail: {arg}")
            raise
            
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def clean_text(self, text):
        """Clean text by removing unwanted characters and normalizing spaces"""
        if not text:
            return ""
            
        # Convert to string if not already
        text = str(text)
        
        # Remove unwanted characters and normalize spaces
        text = re.sub(r'[^\w\s\d:.-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
        
    def extract_score(self, text):
        """Extract score from text"""
        try:
            # Clean up the text first
            text = self.clean_text(text)
            
            # Look for score patterns
            score_patterns = [
                r'(\d+)\s*[-:]\s*(\d+)',  # Basic pattern: X-Y or X:Y
                r'(\d+)\s+[-:]\s+(\d+)',  # Pattern with spaces
                r'(\d+)[-:](\d+)',        # Pattern without spaces
                r'(\d+)\s*τελικό\s*(\d+)',  # Score with "τελικό" (final)
                r'(\d+)\s*σκορ\s*(\d+)',    # Score with "σκορ" (score)
            ]
            
            for pattern in score_patterns:
                match = re.search(pattern, text)
                if match:
                    score1 = int(match.group(1))
                    score2 = int(match.group(2))
                    
                    # Validate scores
                    if score1 >= 0 and score1 <= 20 and score2 >= 0 and score2 <= 20:
                        return score1, score2
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error extracting score: {str(e)}")
            return None
            
    def parse_match_text(self, text):
        """Parse match text to extract teams and scores (robust version)"""
        import re
        try:
            text = self.clean_text(text)
            self.logger.info(f"[parse_match_text] Raw: {text}")

            # Flexible regex: team1 <space> score1 [-:] score2 <space> team2
            pattern = r"([\w\sΑ-Ωα-ω\.\-']+?)\s+(\d+)\s*[-:]\s*(\d+)\s+([\w\sΑ-Ωα-ω\.\-']+)"
            match = re.search(pattern, text)
            if match:
                home_team = match.group(1).strip()
                home_score = int(match.group(2))
                away_score = int(match.group(3))
                away_team = match.group(4).strip()
                # Clean up team names
                home_team = re.sub(r'[^Α-Ωα-ωA-Za-z0-9\s\-\.]', '', home_team).strip()
                away_team = re.sub(r'[^Α-Ωα-ωA-Za-z0-9\s\-\.]', '', away_team).strip()
                if len(home_team) < 2 or len(away_team) < 2:
                    self.logger.warning(f"[parse_match_text] Invalid team names: '{home_team}' vs '{away_team}' in '{text}'")
                    return None
                self.logger.info(f"[parse_match_text] Parsed: {home_team} {home_score}-{away_score} {away_team}")
                return {
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score
                }
            else:
                # Try reversed order (team2 first)
                pattern_rev = r"([\w\sΑ-Ωα-ω\.\-']+)\s+(\d+)\s*[-:]\s*(\d+)"
                match_rev = re.search(pattern_rev, text)
                if match_rev:
                    self.logger.warning(f"[parse_match_text] Only partial match (no away team): '{text}'")
                else:
                    self.logger.warning(f"[parse_match_text] No match for: '{text}'")
                return None
        except Exception as e:
            self.logger.warning(f"[parse_match_text] Error parsing: {text} | {str(e)}")
            return None
            
    def scrape_matches(self, url="https://www.agones.gr/football/matches"):
        """Scrape match data from agones.gr"""
        try:
            self.logger.info(f"Starting scraping from {url}")
            
            # Navigate to the main page and wait for it to load
            self.driver.get(url)
            time.sleep(5)  # Wait for initial load
            
            # Wait for the page to load and scroll multiple times
            scroll_attempts = 10
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            
            for _ in range(scroll_attempts):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            
            # JavaScript: relax filtering, capture more candidates
            script = """
            function findMatchElements() {
                const results = [];
                function getTextContent(element) {
                    return Array.from(element.childNodes)
                        .filter(node => node.nodeType === 3 || node.nodeType === 1)
                        .map(node => node.textContent || node.innerText || '')
                        .join(' ')
                        .replace(/\\s+/g, ' ')
                        .trim();
                }
                function findMatches(root = document) {
                    const elements = root.getElementsByTagName('*');
                    for (const el of elements) {
                        if (el.tagName === 'SCRIPT' || el.tagName === 'STYLE') continue;
                        const text = getTextContent(el);
                        if (!text || text.length < 6) continue;
                        // Relaxed: just look for score pattern
                        const scoreMatch = text.match(/\\b(\\d+)\\s*[-:]\\s*(\\d+)\\b/);
                        if (scoreMatch) {
                            results.push({
                                text: text,
                                rect: el.getBoundingClientRect()
                            });
                        }
                    }
                }
                const sections = [
                    document.querySelector('main'),
                    document.querySelector('#content'),
                    document.querySelector('.matches'),
                    document.querySelector('.results'),
                    document.querySelector('.scores'),
                    ...Array.from(document.querySelectorAll('div[class*="match"]')),
                    ...Array.from(document.querySelectorAll('div[class*="score"]')),
                    ...Array.from(document.querySelectorAll('div[class*="result"]')),
                ];
                for (const section of sections) {
                    if (section) findMatches(section);
                }
                if (results.length === 0) findMatches(document.body);
                return results;
            }
            return findMatchElements();
            """
            
            elements = self.driver.execute_script(script)
            
            if not elements:
                self.logger.error("No elements with scores found")
                return pd.DataFrame()
            
            # Save all raw candidate texts for debugging
            with open('debug_matches_raw.txt', 'w', encoding='utf-8') as f:
                for element in elements:
                    raw_text = element['text'].strip()
                    f.write(raw_text + '\n---\n')
            self.logger.info(f"Saved {len(elements)} raw candidate snippets to debug_matches_raw.txt")
            
            # Process matches
            match_data = []
            seen_matches = set()
            for element in elements:
                try:
                    match_text = element['text'].strip()
                    if not match_text:
                        continue
                    self.logger.info(f"Processing match text:\n{match_text}")
                    # Parse the match text
                    match_info = self.parse_match_text(match_text)
                    if match_info:
                        match_key = f"{match_info['home_team']}_{match_info['away_team']}"
                        if match_key not in seen_matches:
                            seen_matches.add(match_key)
                            # Determine result
                            if match_info['home_score'] > match_info['away_score']:
                                result = 'home'
                            elif match_info['home_score'] < match_info['away_score']:
                                result = 'away'
                            else:
                                result = 'draw'
                            match_data.append({
                                'date': datetime.now().strftime("%Y-%m-%d"),
                                'home_team': match_info['home_team'],
                                'away_team': match_info['away_team'],
                                'home_score': match_info['home_score'],
                                'away_score': match_info['away_score'],
                                'result': result
                            })
                            self.logger.info(f"Successfully processed: {match_info['home_team']} {match_info['home_score']}-{match_info['away_score']} {match_info['away_team']} ({result})")
                except Exception as e:
                    self.logger.warning(f"Failed to process match: {str(e)}")
                    continue
            df = pd.DataFrame(match_data)
            if not df.empty:
                self.logger.info(f"Successfully scraped {len(df)} matches")
                result_counts = df['result'].value_counts()
                self.logger.info("\nResult distribution:")
                self.logger.info(result_counts.to_string())
                min_samples = 2
                if (result_counts >= min_samples).all():
                    self.logger.info(f"All classes have at least {min_samples} samples")
                    df.to_csv('raw_matches.csv', index=False, encoding='utf-8')
                    return df
                else:
                    insufficient_classes = result_counts[result_counts < min_samples].index.tolist()
                    self.logger.error(f"Insufficient samples for classes: {insufficient_classes}")
                    return pd.DataFrame()
            else:
                self.logger.error("No valid match data found")
                return df
        except Exception as e:
            self.logger.error(f"Scraping failed: {str(e)}")
            if hasattr(e, 'msg'):
                self.logger.error(f"Error message: {e.msg}")
            return pd.DataFrame()
        finally:
            try:
                self.driver.quit()
            except Exception as e:
                self.logger.warning(f"Error closing driver: {str(e)}")
                pass

    def save_data(self, df, filename=None):
        """Save scraped data to CSV"""
        if filename is None:
            filename = f"matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            df.to_csv(filename, index=False)
            self.logger.info(f"Data saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save data: {str(e)}")

if __name__ == "__main__":
    scraper = FootballScraper()
    matches_df = scraper.scrape_matches()
    if not matches_df.empty:
        scraper.save_data(matches_df)
