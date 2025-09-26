import asyncio
import asyncpg
import os
from dotenv import load_dotenv


async def main():
    load_dotenv()

    # Database connection parameters
    db_config = {
        "user": os.getenv("PSQL_DB_USERNAME"),
        "password": os.getenv("PSQL_DB_PWD"),
        "host": os.getenv("PSQL_DB_HOSTNAME"),
        "database": os.getenv("PSQL_DB"),
        "port": int(os.getenv("PSQL_DB_PORT", 5432)),
    }

    conn = await asyncpg.connect(**db_config)

    try:
        # Check if user_prefs_api table already exists
        existing_table = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'user_prefs_api'
            );
        """)

        if not existing_table:
            print("Creating user_prefs_api table...")

            create_table_sql = """
                CREATE TABLE user_prefs_api (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    user_vector vector(4096) NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """

            await conn.execute(create_table_sql)
            print("User_prefs_api table created successfully!")

            # Create index on username for faster lookups
            await conn.execute("CREATE INDEX idx_user_prefs_api_username ON user_prefs_api(username)")
            print("Username index created!")

            # Insert two default users with random vectors
            print("Creating default users...")

            # Generate zero vectors as starting point
            user1_vector = [0.0] * 4096
            user2_vector = [0.0] * 4096

            # Convert to PostgreSQL vector format
            user1_vector_str = '[' + ','.join(map(str, user1_vector)) + ']'
            user2_vector_str = '[' + ','.join(map(str, user2_vector)) + ']'

            await conn.execute("""
                INSERT INTO user_prefs_api (username, user_vector) VALUES ($1, $2)
            """, "user1", user1_vector_str)

            await conn.execute("""
                INSERT INTO user_prefs_api (username, user_vector) VALUES ($1, $2)
            """, "user2", user2_vector_str)

            print("Default users created: user1, user2")

        else:
            print("User_prefs_api table already exists.")
            # Update existing users to have zero vectors
            print("Updating existing users to zero vectors...")
            zero_vector_str = '[' + ','.join(['0.0'] * 4096) + ']'

            result = await conn.execute("""
                UPDATE user_prefs_api SET user_vector = $1
            """, zero_vector_str)

            updated_count = int(result.split()[-1])  # Extract count from "UPDATE n"
            print(f"Updated {updated_count} users to zero vectors")

        # Check if user_likes table exists for tracking likes
        existing_likes_table = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'user_likes'
            );
        """)

        if not existing_likes_table:
            print("Creating user_likes table...")

            create_likes_table_sql = """
                CREATE TABLE user_likes (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES user_prefs_api(id),
                    post_id INTEGER,
                    liked_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(user_id, post_id)
                );
            """

            await conn.execute(create_likes_table_sql)
            print("User_likes table created successfully!")

            # Create indexes for faster lookups
            await conn.execute("CREATE INDEX idx_user_likes_user_id ON user_likes(user_id)")
            await conn.execute("CREATE INDEX idx_user_likes_post_id ON user_likes(post_id)")
            print("User_likes indexes created!")

        else:
            print("User_likes table already exists.")
            # Flush all existing likes
            print("Flushing all existing user likes...")
            result = await conn.execute("DELETE FROM user_likes")
            deleted_count = int(result.split()[-1])  # Extract count from "DELETE n"
            print(f"Deleted {deleted_count} existing likes")

        # Display current users
        users = await conn.fetch("SELECT id, username, created_at FROM user_prefs_api ORDER BY id")
        print("\nCurrent users in database:")
        for user in users:
            print(f"  ID: {user['id']}, Username: {user['username']}, Created: {user['created_at']}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())