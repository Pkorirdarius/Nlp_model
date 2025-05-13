import re
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from webdriver_manager.firefox import GeckoDriverManager

def setup_selenium():
    firefox_options = FirefoxOptions()
    firefox_options.add_argument("--headless")  # Run in headless mode
    firefox_options.set_preference("general.useragent.override", 
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:117.0) Gecko/20100101 Firefox/117.0")

    try:
        service = FirefoxService(executable_path=GeckoDriverManager().install())
        driver = webdriver.Firefox(service=service, options=firefox_options)
        print(f"✅ FirefoxDriver setup complete. Version: {driver.capabilities['moz:geckodriverVersion']}")
        return driver

    except Exception as e:
        print(f"❌ Failed to setup FirefoxDriver: {str(e)}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Ensure Firefox is installed and updated")
        print("2. Run: pip install --upgrade selenium webdriver-manager")
        print("3. Try running in non-headless mode by removing '--headless'")
        raise

# Improved content extraction with better error handling
def scrape_who_gmo_faq():
    url = "https://www.who.int/news-room/q-a-detail/food-genetically-modified"
    driver = setup_selenium()
    
    try:
        driver.get(url)
        
        # Wait for main content with increased timeout
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "article, .sf-content-block, .q-a-detail"))
        )

        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Try multiple selectors for main content
        selectors = [
            {"name": "div", "class": "sf-content-block"},
            {"name": "article"},
            {"name": "main"},
            {"name": "div", "class": "q-a-detail"}
        ]
        
        main_content = None
        for selector in selectors:
            main_content = soup.find(selector["name"], class_=selector.get("class"))
            print(f"Trying selector: {selector} -> Found: {bool(main_content)}")
            if main_content:
                break

                
        if not main_content:
            raise ValueError("Main content container not found - page structure may have changed")
        
        # Improved Q&A extraction
        qa_pairs = []
        current_question = None
        current_answer = []

        elements = main_content.find_all(["h2", "h3", "h4", "p", "div"])

        for element in elements:
            text = element.get_text(strip=True)
            if not text:
                continue

        if element.name in ["h2", "h3", "h4"] and len(text) < 200:
        # Treat as a question if it’s short and a heading
            if current_question and current_answer:
                qa_pairs.append({
                    "question": current_question,
                    "answer": " ".join(current_answer).strip()
                })
            current_question = text
            current_answer = []
        elif current_question:
            current_answer.append(text)

        # Capture any remaining pair
        if current_question and current_answer:
            qa_pairs.append({
                "question": current_question,
             "answer": " ".join(current_answer).strip()
            })

        return qa_pairs
    
    except Exception as e:
        print(f"Error during scraping: {e}")
        return None
    finally:
        driver.quit()

# Enhanced rebuttal generation with better text cleaning
def generate_rebuttals(qa_pairs):
    if not qa_pairs:
        return None
        
    misconception_map = {
        "GMOs cause cancer": {
            "keywords": [r'cancer', r'carcinogen', r'oncogen'],
            "type": "health"
        },
        "GMOs are unsafe": {
            "keywords": [r'safe', r'safety', r'risk', r'unsafe', r'hazard'],
            "type": "safety"
        },
        "GMOs cause diseases": {
            "keywords": [r'disease', r'illness', r'health effect', r'pathogen'],
            "type": "health"
        },
        "GMOs are not regulated": {
            "keywords": [r'regulat', r'approval', r'assessment', r'authorities', r'standard'],
            "type": "regulation"
        },
        "GMOs lead to infertility": {
            "keywords": [r'infertility', r'reproductive', r'fertility', r'sterility'],
            "type": "health"
        }
    }
    
    data = []
    
    for claim, claim_data in misconception_map.items():
        pattern = re.compile('|'.join(claim_data['keywords']), re.IGNORECASE)
        
        for qa in qa_pairs:
            if pattern.search(qa['question']) or pattern.search(qa['answer']):
                # Improved text cleaning
                clean_answer = re.sub(r'\s+', ' ', qa['answer'])
                clean_answer = re.sub(r'\[.*?\]', '', clean_answer)  # Remove citations
                clean_answer = clean_answer[:800].rsplit(' ', 1)[0] + '...'  # Clean truncation
                
                data.append({
                    "Claim_text": claim,
                    "Claim_type": claim_data["type"],
                    "Rebuttal_text": clean_answer,
                    "Source_abbreviation": "WHO",
                    "Tone": "Scientific",
                    "Label": "false",
                    "Rebuttal_strength": 5
                })
                break  # Use first relevant match
    
    return data if data else None

# Main execution with better error handling
if __name__ == "__main__":
    try:
        print("🔄 Scraping WHO GMO FAQ page...")
        qa_pairs = scrape_who_gmo_faq()
        
        if not qa_pairs:
            print("⚠️ No Q&A pairs found. Possible reasons:")
            print("- Page structure changed")
            print("- Content not loading properly")
            print("- Website blocking requests")
        else:
            print(f"✅ Found {len(qa_pairs)} Q&A pairs")
            rebuttals = generate_rebuttals(qa_pairs)
            
            if rebuttals:
                df = pd.DataFrame(rebuttals)
                # Improved CSV output
                df.to_csv("gmo_rebuttals_selenium.csv", 
                         index=False, 
                         encoding='utf-8-sig',
                         quoting=pd.io.common.csv.QUOTE_NONNUMERIC)
                print(f"✅ Saved {len(df)} rebuttals to gmo_rebuttals_selenium.csv")
                print("Sample rebuttal:")
                print(df.iloc[0]["Rebuttal_text"][:200] + "...")
            else:
                print("⚠️ No matching rebuttals found. Try:")
                print("- Adjusting keywords in misconception_map")
                print("- Checking the scraped Q&A pairs in who_gmo_faq.csv")
                pd.DataFrame(qa_pairs).to_csv("who_gmo_faq.csv", index=False)
    
    except Exception as e:
        print(f"❌ Critical error: {e}")
        print("Try:")
        print("1. Updating Chrome browser")
        print("2. Checking internet connection")
        print("3. Running in non-headless mode for debugging")