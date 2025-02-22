from pymongo import MongoClient

# Replace <username> and <password> with your MongoDB credentials
MONGO_URI = "mongodb+srv://nikhathmahammad12:AN2bMxZTXGndN4or@cluster0.asogv.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(MONGO_URI)
db = client["Drowsi"]  # Database Name
users_collection = db["Drowsi"]  # Collection Name
