import lancedb

# Connect to the LanceDB
db = lancedb.connect("./db")

# Open the listings table
table = db.open_table("listings")

# Get the first 5 rows as Arrow table, then convert to dict
arrow_table = table.to_arrow().slice(0, 5).to_pydict()

# Iterate over each listing
for i in range(len(arrow_table["title"])):
    print("Title:", arrow_table["title"][i])
    print("Location:", arrow_table["location"][i])
    print("Price: $", arrow_table["price"][i])
    print("Bedrooms:", arrow_table["number_of_bedrooms"][i])
    print("Bathrooms:", arrow_table["number_of_bathrooms"][i])
    print("Square Feet:", arrow_table["square_feet"][i])
    print("Amenities:", arrow_table["amenities"][i])
    print("Neighborhood:", arrow_table["neighborhood"][i])
    print("Description:", arrow_table["description"][i])
    print("-" * 40)
