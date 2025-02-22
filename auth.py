import bcrypt
from database import users_collection

def register_user(email, password):
    """Register a new user by hashing their password and storing it in MongoDB."""
    if users_collection.find_one({"email": email}):
        return "User already exists!"
    
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users_collection.insert_one({"email": email, "password": hashed_pw})

    return "User registered successfully!"

def login_user(email, password):
    """Authenticate user by checking their hashed password."""
    user = users_collection.find_one({"email": email})

    if user and bcrypt.checkpw(password.encode('utf-8'), user["password"]):
        return "Login successful!"
    else:
        return "Invalid email or password."

# Example Usage
if __name__ == "__main__":
    print(register_user("testuser@example.com", "mypassword"))
    print(login_user("testuser@example.com", "mypassword"))
