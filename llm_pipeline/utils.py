import logging
import random
import re
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fallback values
CITIES = ["Berlin", "Munich", "Hamburg", "Cologne", "Frankfurt"]
# NEIGHBORHOODS = ["Charlottenburg", "Prenzlauer Berg", "Altstadt", "Schwabing", "Sachsenhausen"]


# TITLES = ["Spacious City Apartment", "Modern Family Home", "Cozy Urban Flat", "Luxury Penthouse", "Charming Studio"]
# AMENITIES = ["Balcony", "Garage", "Garden", "Elevator", "Fireplace", "Gym", "Swimming Pool"]


def pick_random(lst: list) -> any:
    """Select a random element from a list.
    If the list is empty, return None.
    Args:
        lst (list): The list to pick from.
    Returns:
        Any: A random element from the list, or None if the list is empty.
    """
    return random.choice(lst)


def safe_int(val, default=1) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        logger.warning(f"⚠️ Could not parse int from: {val}, using default: {default}")
        return default


def safe_float(val, default=1000.0) -> float:
    try:
        return float(re.sub(r"[^\d.]", "", str(val)))
    except Exception as excpt:
        logger.warning(f"⚠️ Could not parse float from: {val}, using default: {default} | Error: {excpt}")
        return default


def parse_price(val) -> float:
    return safe_float(val, default=random.randint(100_000, 900_000))


def fallback_string(val, fallback_list) -> str:
    return str(val) if val else pick_random(fallback_list)


def fallback_int(val, default=None) -> int:
    if default is None:
        default = random.randint(1, 4)
    return safe_int(val, default=default)


def get_fallback_description(neighborhood=None) -> str:
    n = neighborhood
    return f"The {n} area is known for its charm, amenities, and accessibility."


def setup_llm(openai_api_key: str) -> tuple:
    """Initialize the LLM, embeddings, and memory for the pipeline.
    :param openai_api_key: Your OpenAI API key.
    :return: Tuple of (embedding, llm, memory)
    """
    embedding = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return embedding, llm, memory


def augment_listing_with_preferences(listing: dict, preferences: str, llm: ChatOpenAI) -> str:
    """
    Enhance listing with user preferences via structured prompt.
    :param listing: The real estate listing data as a dictionary.
    :param preferences: User preferences for the listing.
    :param llm: The LLM instance to use for generating the augmented description.
    :return: The refined description of the listing.
    """
    print("\n \n \n Augmenting listing with preferences... \n \n \n")
    title = listing.get("title")
    location = listing.get("location")
    bedrooms = listing.get("number_of_bedrooms")
    bathrooms = listing.get("number_of_bathrooms")
    area = listing.get("square_feet")
    price = listing.get("price")
    amenities = listing.get("amenities")

    description = listing.get("page_content")
    neighborhood_desc = listing.get("neighborhood_description")

    # Build augmented description
    raw_augmented_description = (
        f"Welcome to this cozy {bedrooms}-bedroom, {bathrooms}-bathrooms home "
        f"that could be perfect for your family in {location}! "
        f"It offers approximately {area} square feet of living space, priced at ${int(price):,}. "
        f"{description} {neighborhood_desc}"
        f"This fits your preference for {preferences} with the following amenities available:\n\n"
        "".join(
            f"- {amenity}\n" for amenity in amenities
        )
    )

    # Prompt the LLM
    prompt = (
        "Refine the following real estate listing description to sound more professional and appealing. "
        "Keep all the factual information intact:\n\n"
        f"{raw_augmented_description}"
    )

    try:
        refined = llm.predict(prompt)
        return refined.strip()
    except Exception as e:
        import logging
        logging.warning(f"⚠️ LLM call failed in augment_listing_with_preferences: {e}")
        return raw_augmented_description
