# Hotel Review AI Manager

An intelligent automation system for hotel management that combines local **Natural Language Processing (NLP)** with **Generative AI**. This tool analyzes guest reviews to detect sentiment and specific issues, then generates professional responses.



## System Architecture

The application uses a hybrid approach, combining high-speed machine learning with deep learning context understanding:

1. **Sentiment Analysis (BERT)**
   - **Model:** `bert-base-uncased` (Fine-tuned).
   - **Task:** Accurately determines if a review is positive or negative.
   - **Why:** BERT understands the context of words in a sentence, making it excellent for detecting emotions.

2. **Category Classification (XGBoost)**
   - **Model:** Extreme Gradient Boosting.
   - **Task:** Classifies reviews into 6 key categories:
     - `Cleanliness`
     - `Service & Management`
     - `Amenities & Room`
     - `Food & Breakfast`
     - `Location & View`
     - `Price & Value`
   - **Technique:** Uses TF-IDF with N-grams (1-3) to understand phrases like "not very clean."

3. **Response Generation (Gemini AI)**
   - **Model:** `gemini-2.5-flash`.
   - **Task:** Generates context-aware, polite, and professional replies based on the detected sentiment and category.



## Workflow Example

**User Input:** > "The breakfast was delicious, but the WiFi signal in our room was extremely weak."

**AI Analysis:**
* **Sentiment:** `NEGATIVE` (prioritizes the complaint).
* **Category:** `Amenities & Room`.
* **Response:** *"We are delighted to hear you enjoyed our breakfast! However, we sincerely apologize for the WiFi issues in your room; we are currently upgrading our routers to ensure a better connection for our guests."*

## Getting Started

### Prerequisites
- Python 3.9+
- Gemini API Key (stored in `.env` file)

### Dataset 
- Download the **TripAdvisor Hotel Reviews** dataset from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/tripadvisor-hotel-reviews)
- Place the `tripadvisor_hotel_reviews.csv` file in the root directory of the project

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/alinakopasava/hotel-ai-manager.git](https://alinakopasava/hotel-ai-manager.git)
    ```
    ```bash
   cd hotel-ai-manager
    ```

2. Create virtual environment: 
    ```bash
   python -m venv venv
    ```
3. Activate the environment:
    ```bash
   venv\Scripts\activate
    ```
4. Install dependencies 
    ```bash
   pip install -r requirements.txt
    ```
5. Create a .env file in the root directory and add your API Key:
    ```
   GEMINI_API_KEY=api_key
    ```

### Usage 
Run the scripts in this order:  
   `data.py `  
   `labeling.py`  
   `categories_xgboost.py`  
   `sentiment_bert.py`  

### Running the service
```
streamlit run app.py 
```