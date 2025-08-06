import os
import json
import streamlit as st
import pandas as pd
from llm_pipeline.listing_generator import RealEstateIndexer

from llm_pipeline.utils import augment_listing_with_preferences, setup_llm, CITIES
from llm_pipeline.logger import setup_logger


class RealEstateApp:
    """A Streamlit application for managing real estate listings,
    allowing users to search for properties based on preferences or generate new listings.
    This application uses a language model to enhance the user experience by generating listings
    and providing personalized recommendations based on user inputs.
    """

    def __init__(self):
        self.indexer = None
        self.agent = None
        self.vectorstore = None
        self.db = None
        self.csv_file = None
        self.memory = None
        self.llm = None
        self.embedding = None
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.setup_llm()
        self.setup_RealEstateIndexer()
        self.logger = setup_logger()

    def setup_llm(self):
        """Sets up the language model and embedding for the application."""
        self.embedding, self.llm, self.memory = setup_llm(openai_api_key=self.openai_api_key)

    def setup_RealEstateIndexer(self):
        """Initializes the RealEstateIndexer with the database path, table name, and output CSV file."""
        self.indexer = RealEstateIndexer(
            db_path="./db",
            table_name="listings",
            output_csv="./outputs/listings.csv"
        )

    @staticmethod
    def parse_preferences(user_inputs_dict: dict) -> str:
        """
        Parses user inputs into a structured string for the language model.
        :param user_inputs_dict:
        :return: str
            A structured string that summarizes the user's preferences for real estate listings.
        """
        prompt_parts = []
        if user_inputs_dict.get("square_feet"):
            prompt_parts.append(f"I want a house of around {user_inputs_dict['square_feet']} sq ft.")
        if user_inputs_dict.get("city"):
            prompt_parts.append(f"My preferred city is {user_inputs_dict['city']}.")
        if user_inputs_dict.get("budget"):
            prompt_parts.append(f"My budget is around {user_inputs_dict['budget']}.")
        if user_inputs_dict.get("features"):
            prompt_parts.append(f"I'm looking for features like: {user_inputs_dict['features']}.")
        if user_inputs_dict.get("amenities"):
            prompt_parts.append(f"I want amenities like: {', '.join(user_inputs_dict['amenities'])}.")
        if user_inputs_dict.get("neighborhoods"):
            prompt_parts.append(f"My preferred neighborhoods are: {', '.join(user_inputs_dict['neighborhoods'])}.")
        return "\n".join(prompt_parts)

    def get_top_k_similar_listings(self, user_query: str, k=3) -> list:
        """
        Retrieves the top k similar listings based on the user's query.
        This method uses the vectorstore to perform a similarity search.
        It returns a list of documents that match the user's preferences.
        This is useful for finding listings that closely match the user's requirements.
        The method assumes that the vectorstore has been properly initialized and contains the necessary data.
        It is designed to work with the RealEstateIndexer class, which manages the listings and their embeddings.
        :param user_query: str
            The user's query or preferences as a string.
        :param k: int
            The number of top similar listings to retrieve.
        :raises Exception: If the vectorstore is not initialized or if the search fails.
        :return: list
            A list of documents that are similar to the user's query.
        This list contains the top k listings that match the user's preferences.
        Each document in the list includes metadata and content that can be used to display the listings.
        """
        vectorstore = self.indexer.return_vectorstore()
        return vectorstore.similarity_search(user_query, k=k)

    @staticmethod
    def rerank_listings(listings: list, user_inputs: dict) -> list:
        """
        Reranks the listings based on user inputs.
        This method takes a list of listings and user inputs, scores each listing based on the user's preferences,
        and returns the top 3 listings sorted by their scores.
        :param listings: list
            A list of listings to be reranked. Each listing is expected to have a metadata field
        :param user_inputs: dict
            A dictionary containing user preferences such as city, budget, square footage, features, and amenities
        :return: list
            A list of the top 3 listings, each represented as a tuple of (score, listing).
        """
        def score_listing(individual_listing):
            score = 0
            metadata = individual_listing.metadata

            # Hard skip if not from the selected city
            if user_inputs.get("city") and metadata.get("location"):
                if metadata["location"].lower() != user_inputs["city"].lower():
                    return 0

            # 1. City match (extra optional boost if needed)
            score += 5  # We already hard-filtered, so this is just to influence sorting

            # 2. Budget
            if user_inputs.get("budget") and metadata.get("price"):
                try:
                    budget = float(user_inputs["budget"])
                    price = float(metadata["price"])
                    diff = abs(budget - price)
                    score += max(0, 3 - diff / budget)
                except:
                    pass

            # 3. Square footage
            if user_inputs.get("square_feet") and metadata.get("square_feet"):
                try:
                    desired = float(user_inputs["square_feet"])
                    actual = float(metadata["square_feet"])
                    diff = abs(desired - actual)
                    score += max(0, 2 - diff / desired)
                except:
                    pass

            # 4. Bedroom count
            if "number_of_bedrooms" in metadata and user_inputs.get("features"):
                if str(metadata["number_of_bedrooms"]) in user_inputs["features"]:
                    score += 1

            # 5. Bathroom count
            if "number_of_bathrooms" in metadata and user_inputs.get("features"):
                if str(metadata["number_of_bathrooms"]) in user_inputs["features"]:
                    score += 1

            # 6. Amenities
            if "amenities" in metadata and user_inputs.get("amenities"):
                if isinstance(metadata["amenities"], list):
                    matched = set(user_inputs["amenities"]).intersection(set(metadata["amenities"]))
                    score += len(matched) * 0.5

            return score

        # Attach scores and sort
        scored = [(score_listing(doc), doc) for doc in listings]
        ranked = sorted(scored, key=lambda x: x[0], reverse=True)
        print("Ranked Listings:", ranked[:3])
        return ranked[:3]  # list of (score, doc)

    def run_ui(self) -> None:
        """
        Runs the Streamlit UI for the real estate application.
        :return: None
        This method initializes the Streamlit application, sets up the user interface,
        and handles user interactions for searching listings or generating new listings.
        """
        st.title("🏘️ Real Estate Listing Assistant")

        mode = st.selectbox("Choose mode", ["🔍 Search Listings", "🛠️ Generate New Listing"])

        if mode == "🔍 Search Listings":
            st.subheader("Search Listings")

            with st.form("search_form"):
                sqft = st.text_input("How big do you want your house to be (in sq feet)?")
                city = st.selectbox("What is the city you are interested in?", CITIES)
                budget = st.text_input("What is your budget?")
                features = st.text_input("How many bathrooms, bedrooms and other interesting features do you want?")
                amenities = st.text_input("Which amenities would you like?")
                neighborhoods = st.text_input("What are your preferred neighborhoods?")

                submitted = st.form_submit_button("Search")
                if submitted:
                    raw_inputs = {
                        "square_feet": sqft,
                        "city": city,
                        "budget": budget,
                        "features": features,
                        "amenities": amenities,
                        "neighborhoods": neighborhoods
                    }

                    # Remove keys where value is None, empty string, or empty list
                    user_inputs_dict = {k: v for k, v in raw_inputs.items() if v not in [None, "", []]}
                    self.search_listings(user_inputs_dict)

        elif mode == "🛠️ Generate New Listing":
            st.subheader("Generate a New Listing")

            with st.form("generate_form"):
                sqft = st.text_input("Enter desired square footage:")
                budget = st.text_input("What is the cost of the new listing?")
                features = st.text_input("Describe desired number of bedrooms, bathrooms, or any unique features:")
                amenities = st.text_input("Which amenities would you like to add for this listing?")
                city = st.selectbox("Select a city", CITIES)
                neighborhood = st.text_input("Provide a neighborhood")

                submitted = st.form_submit_button("Generate")
                if submitted:
                    raw_inputs = {
                        "square_feet": sqft,
                        "city": city,
                        "budget": budget,
                        "features": features,
                        "amenities": amenities,
                        "neighborhoods": neighborhood
                    }

                    # Remove keys where value is None, empty string, or empty list
                    user_inputs_dict = {k: v for k, v in raw_inputs.items() if v not in [None, "", []]}
                    self.generate_listing(user_inputs_dict)

        if st.checkbox("Show all stored listings (latest first)"):
            self.display_all_listings()

    def search_listings(self, user_inputs: dict) -> None:
        """
        Searches for real estate listings based on user preferences.
        This method takes user inputs, parses them into a format suitable for the language model,
        and retrieves the top k similar listings from the vectorstore.
        It displays the results in the Streamlit UI, including enriched descriptions and metadata.
        :param user_inputs: dict
            A dictionary containing user preferences such as city, budget, square footage, features, and amenities
        :return: None
        """

        user_pref = self.parse_preferences(user_inputs)
        print("User Preferences:", user_pref)
        st.session_state["latest_pref"] = user_pref
        with st.spinner("Searching listings..."):
            try:
                # Get the top k similar listings based on user preferences
                raw_results = self.get_top_k_similar_listings(user_pref, k=10)
                # Rerank the results based on user inputs
                results = self.rerank_listings(raw_results, user_inputs)
                if results:
                    st.subheader("🔍 Top Matching Listings")
                    for i, (score, doc) in enumerate(results, 1):
                        metadata = doc.metadata
                        original_desc = doc.page_content
                        augmented_desc = augment_listing_with_preferences(metadata, user_pref, self.llm)

                        st.write(f"### 🏡 Listing #{i}")
                        st.write("**Description:**", augmented_desc)
                        # Display enriched listing metadata
                        print("Listing Metadata:", metadata)

                        # Mandatory fields
                        st.markdown(f"**🏷️ Title:** {metadata.get("title")}")
                        st.markdown(f"**📍 Location:** {metadata.get('location', 'N/A')}")
                        st.markdown(
                            f"**🛏️ Bedrooms:** {metadata.get('number_of_bedrooms', 'N/A')} | 🛁 Bathrooms:** "
                            f"{metadata.get('number_of_bathrooms', 'N/A')}")
                        st.markdown(
                            f"**📐 Square Feet:** {metadata.get('square_feet', 'N/A')} | "
                            f"💰 Price:** ${metadata.get('price', 'N/A'):,}")
                        st.markdown(f"**🏘️ Neighborhood:** {metadata.get('neighborhood', 'N/A')}")

                        # Handle amenities
                        amenities = metadata.get('amenities', [])
                        if isinstance(amenities, str):
                            try:
                                amenities = json.loads(amenities)
                            except json.JSONDecodeError:
                                amenities = [amenities]
                        if amenities:
                            st.markdown(f"**✅ Amenities:** {', '.join(amenities)}")

                        st.write("---")
                else:
                    st.warning("No listings matched your preferences.")
            except Exception as e:
                st.error(f"Search failed: {e}")

    def generate_listing(self, user_inputs: dict) -> None:
        """
        Generates a new real estate listing based on user preferences.
        This method takes user inputs, parses them into a format suitable for the language model,
        :param user_inputs: list[str]
            A list of user inputs containing preferences for the new listing.
        :return: None
        """
        user_pref = self.parse_preferences(user_inputs)
        with st.spinner("Generating listing..."):
            try:
                listing = self.indexer.run_generation_loop(count=1, minimal_input=user_pref)
                if listing:
                    self.indexer.index_and_save(listings=listing)

                    st.write(f"### 🏡 New Listing: {listing[0].get('title', 'N/A')}")
                    # st.write("**Original Description:**", original_desc)
                    st.write("**Description:**", listing[0].get('description', 'N/A'))
                    st.markdown(f"**📍 Location:** {listing[0].get('location', 'N/A')}")
                    st.markdown(
                        f"**🛏️ Bedrooms:** {listing[0].get('number_of_bedrooms', 'N/A')} | 🛁 Bathrooms:** "
                        f"{listing[0].get('number_of_bathrooms', 'N/A')}")
                    st.markdown(
                        f"**📐 Square Feet:** {listing[0].get('square_feet', 'N/A')} | "
                        f"💰 Price:** ${listing[0].get('price', 'N/A'):,}")
                    st.markdown(f"**🏘️ Neighborhood:** {listing[0].get('neighborhood', 'N/A')}")

                    # Handle amenities
                    amenities = listing[0].get('amenities', [])
                    if isinstance(amenities, str):
                        try:
                            amenities = json.loads(amenities)
                        except json.JSONDecodeError:
                            amenities = [amenities]
                    if amenities:
                        st.markdown(f"**✅ Amenities:** {', '.join(amenities)}")

                    st.write("---")

            except Exception as e:
                st.error(f"Error parsing response: {e}")
                st.code(listing)

    def display_all_listings(self) -> None:
        """
        Displays all real estate listings stored in the LanceDB database.
        This method retrieves the listings from the LanceDB database, processes them to remove the vector column
        and expands the metadata into separate columns. It then displays the latest listings in a Streamlit dataframe.
        :raises Exception: If there is an error retrieving or processing the listings.
        :return:
        """
        try:
            table = self.indexer.db.open_table("listings")
            df = table.to_pandas()

            # Drop the vector column
            df = df.drop(columns=["vector"], errors="ignore")

            # Expand metadata dict into columns
            metadata_df = pd.json_normalize(df["metadata"])

            print("Metadata DataFrame:", metadata_df.head(5))

            # Combine metadata with the description (page_content)
            df_combined = pd.concat([df["text"], metadata_df], axis=1)
            df_combined.rename(columns={"text": "description"}, inplace=True)

            # Reverse DataFrame to show latest first
            df_combined = df_combined.iloc[::-1]

            # Show only the latest 10
            st.dataframe(df_combined.head(10))

        except Exception as e:
            st.error(f"Failed to load listings from LanceDB: {e}")


if __name__ == "__main__":
    app = RealEstateApp()
    app.run_ui()
