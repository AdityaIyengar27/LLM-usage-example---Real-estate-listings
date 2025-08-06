from llm_pipeline.listing_generator import RealEstateIndexer
from llm_pipeline.logger import setup_logger

# Initialize the logger
logger = setup_logger()


def main() -> None:
    """
    Main function to run the real estate listings generation and indexing pipeline.
    :return: None
    """
    try:
        logger.info("🚀 Starting listing generation and indexing pipeline...")

        # Initialize the indexer
        indexer = RealEstateIndexer(
            db_path="./db",
            table_name="listings",
            output_csv="./outputs/listings.csv"
        )

        # Generate and index listings
        listings = indexer.run_generation_loop(count=50)
        if listings:
            indexer.index_and_save(listings=listings, create_table=True)
        else:
            logger.warning("⚠️ No listings generated.")

        logger.info("✅ Pipeline completed successfully.")

    except Exception as e:
        logger.exception(f"❌ Pipeline failed due to: {e}")


if __name__ == "__main__":
    main()
