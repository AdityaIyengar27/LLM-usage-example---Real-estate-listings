# LLM-usage-example---Real-estate-listings
## 🏡 Real Estate Listing App (with LLM + LanceDB)
This is an AI-powered real estate app that enables users to:
 - Generate realistic property listings using a Language Model (LLM),
 - Store and search listings via a semantic vector database (LanceDB),
 - Search listings based on user preferences (e.g., city, price, square footage),
 - Rerank results based on business priorities like location, budget, etc.
 - Built with Streamlit, LangChain, OpenAI, and LanceDB.

## Features
### Generate property listings using LLMs
- Generate realistic property listings using a Language Model (LLM)
- Generate listings from a script (`generate_listings_and_index.py`)
- Generate listings from a Streamlit app using a form and then filling in using the LLM
- LLM fills in missing fields like description, amenities, and metadata.
- Stores listings in both CSV and LanceDB for persistence and fast retrieval.

### Search Listings
- Search listings based on user preferences (e.g., city, price, square footage, etc.)
- Top 20 similar listings are retrieved via LanceDB vector search.
- Reranking logic prioritizes location, then price, size, bedrooms, bathrooms, and amenities.
- Top 3 reranked listings are displayed in the UI.

### LLM-Powered Description Augmentation
- The retrieved listing descriptions are enhanced to match user preferences using the LLM.

### Project Structure
```
├── app.py                          # Streamlit app for user interface
├── generate_listings_and_index.py  # Script to generate and index listings
├── requirements.txt                # Python dependencies
├── outputs/
│   └── listings.csv                # Generated property listings in CSV format
├── db/
│   └── <db files>                  # LanceDB database files
├── README.md                       # Project documentation
├── .env                            # Environment variables (OpenAI API key, etc.)
└──list_entries.py
```

## How it works
1. **Source Environment Variables**: Load environment variables from `.env` file.
   - For this project, you need to set the `OPENAI_API_KEY` variable in the `.env` file. Set your OpenAI API key there.
   - To execute the code, you can use the command `source .env` in your terminal.
   ```bash
   source .env
    ```
2. **Generate Listings**: Run the `generate_listings_and_index.py` script to generate property listings and index them in LanceDB.
    - This script uses the OpenAI API to generate realistic property listings and stores them in both a CSV file and LanceDB.
    - It generates 100 listings by default, but you can adjust the `num_listings` variable in the script.
    - Embeds them using OpenAI embeddings.
    - Stores them in LanceDB (`db/`) and `listings.csv`.
   ```bash
    python generate_listings_and_index.py
    ```
3. **Run the Streamlit App**: Start the Streamlit app to interact with the listings.
    - The app allows users to search for listings based on their preferences and rerank them.
    - It also allows users to generate new listings using a form and the LLM.
    - Preferences are parsed into a dict.
    - Top 20 similar listings are retrieved from LanceDB.
    - Listings are reranked by city > price > square_feet > bedrooms > bathrooms > amenities.
    - Top 3 are shown to the user with LLM-augmented descriptions.
   ```bash
    streamlit run app.py
    ```
   
### Reranking logic
Listings are reranked based on a custom scoring function:

| **Priority**     | **Weight / Logic**                                              |
|------------------|------------------------------------------------------------------|
| **City**         | Must match (hard filter, +5 points if exact match)              |
| **Price**        | Closer match = higher score: `score += max(0, 3 - diff/budget)` |
| **Square Feet**  | Closer match = higher score: `score += max(0, 2 - diff/desired)`|
| **Bedrooms**     | Match count from features list: `+1` per match                  |
| **Bathrooms**    | Match count from features list: `+1` per match                  |
| **Amenities**    | `+0.5` for each matching amenity                                |

## Setup Instructions
1. **Create a virtual environment** (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
2. **Source Environment Variables**
```bash
source .env
```
3. **Install Dependencies**
```bash
pip install -r requirements.txt
```
4. **Generate Listings and Index**
```bash
python generate_listings_and_index.py
```
5. **Run the Streamlit App**
```bash
streamlit run app.py
```

### Additional Notes
To test if the LanceDB database is working correctly, you can run the `list_entries.py` script. This will list 5 
entries in the LanceDB database and print them to the console.
```bash
python list_entries.py
```
