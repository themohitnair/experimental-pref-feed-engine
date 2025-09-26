import asyncio
import asyncpg
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


def load_qwen_model():
    print("Loading Qwen3-Embedding-8B model in 8-bit...")
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-8B",
        model_kwargs={"load_in_8bit": True, "device_map": "auto"},
    )
    return model


def generate_embedding(model, title, description):
    combined_text = f"{title or ''} {description or ''}".strip()
    if not combined_text:
        return None

    embedding = model.encode(combined_text, convert_to_numpy=True)
    return embedding.tolist()


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
        # First, get the structure of social_search_2 table
        print("Getting structure of social_search_2 table...")
        table_info = await conn.fetch("""
            SELECT column_name, data_type, character_maximum_length, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = 'social_search_2'
            ORDER BY ordinal_position;
        """)

        if not table_info:
            print("Error: social_search_2 table not found!")
            return

        # Check if social_search_prefs already exists
        existing_table = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'social_search_prefs'
            );
        """)

        if not existing_table:
            print("Creating social_search_prefs table...")
            print("Table structure from social_search_2:")
            for col in table_info:
                print(f"  {col['column_name']}: {col['data_type']}")

            # Build CREATE TABLE statement based on social_search_2 structure
            columns = []
            excluded_columns = [
                "canonical_url",
                "vsearch",
                "minilm_vectors",
                "blip_vector",
            ]

            for col in table_info:
                col_name = col["column_name"]

                # Skip excluded columns
                if col_name in excluded_columns:
                    continue

                data_type = col["data_type"]
                max_length = col["character_maximum_length"]
                is_nullable = col["is_nullable"]
                default_val = col["column_default"]

                # Handle reserved keywords by quoting them
                if col_name.upper() in [
                    "USER",
                    "ORDER",
                    "GROUP",
                    "SELECT",
                    "FROM",
                    "WHERE",
                ]:
                    col_name = f'"{col_name}"'

                # Handle different data types
                if data_type == "character varying" and max_length:
                    col_def = f"{col_name} VARCHAR({max_length})"
                elif data_type == "text":
                    col_def = f"{col_name} TEXT"
                elif data_type == "integer":
                    col_def = f"{col_name} INTEGER"
                elif data_type == "bigint":
                    col_def = f"{col_name} BIGINT"
                elif data_type == "timestamp without time zone":
                    col_def = f"{col_name} TIMESTAMP"
                elif data_type == "timestamp with time zone":
                    col_def = f"{col_name} TIMESTAMPTZ"
                elif data_type == "boolean":
                    col_def = f"{col_name} BOOLEAN"
                elif data_type == "numeric":
                    col_def = f"{col_name} NUMERIC"
                elif data_type == "real":
                    col_def = f"{col_name} REAL"
                elif data_type == "double precision":
                    col_def = f"{col_name} DOUBLE PRECISION"
                elif data_type == "jsonb":
                    col_def = f"{col_name} JSONB"
                elif data_type == "json":
                    col_def = f"{col_name} JSON"
                elif data_type == "ARRAY":
                    col_def = f"{col_name} TEXT[]"
                else:
                    col_def = f"{col_name} TEXT"

                if is_nullable == "NO":
                    col_def += " NOT NULL"

                if default_val and "nextval" not in default_val:
                    col_def += f" DEFAULT {default_val}"

                columns.append(col_def)

            # Add the new qwen_vector column
            columns.append("qwen_vector vector(4096)")

            create_table_sql = f"""
                CREATE TABLE social_search_prefs (
                    {", ".join(columns)}
                );
            """

            print(f"CREATE TABLE SQL: {create_table_sql}")
            await conn.execute(create_table_sql)
            print("social_search_prefs table created successfully!")
        else:
            print("social_search_prefs table already exists.")
            # Check if qwen_vector column exists and has correct type
            qwen_col = await conn.fetchval("""
                SELECT data_type FROM information_schema.columns
                WHERE table_name = 'social_search_prefs' AND column_name = 'qwen_vector'
            """)

            if qwen_col != "USER-DEFINED":
                print("Updating qwen_vector column to vector(4096) type...")
                if qwen_col:
                    # Column exists but wrong type, drop and recreate
                    await conn.execute(
                        "ALTER TABLE social_search_prefs DROP COLUMN qwen_vector"
                    )
                # Add the vector column
                await conn.execute(
                    "ALTER TABLE social_search_prefs ADD COLUMN qwen_vector vector(4096)"
                )
                print("qwen_vector column updated successfully!")

        # Check current count in social_search_prefs
        current_count = await conn.fetchval("SELECT COUNT(*) FROM social_search_prefs")
        print(f"Current records in social_search_prefs: {current_count}")

        # Check if we need to copy data
        if current_count == 0:
            print("Copying last 700k posts from social_search_2...")

            # First, let's check what columns exist and find the time column
            time_column = None
            for col in table_info:
                col_name = col["column_name"].lower()
                if "time" in col_name or "date" in col_name or "created" in col_name:
                    time_column = col["column_name"]
                    break

            if not time_column:
                print(
                    "Warning: Could not identify time column. Using first column for ordering."
                )
                time_column = table_info[0]["column_name"]

            print(f"Using column '{time_column}' for time-based ordering")

            # Copy data with LIMIT 700000 ordered by time DESC
            excluded_columns = [
                "canonical_url",
                "vsearch",
                "minilm_vectors",
                "blip_vector",
            ]
            column_names = []
            for col in table_info:
                col_name = col["column_name"]
                # Skip excluded columns
                if col_name in excluded_columns:
                    continue
                if col_name.upper() in [
                    "USER",
                    "ORDER",
                    "GROUP",
                    "SELECT",
                    "FROM",
                    "WHERE",
                ]:
                    col_name = f'"{col_name}"'
                column_names.append(col_name)
            columns_str = ", ".join(column_names)

            copy_sql = f"""
                INSERT INTO social_search_prefs ({columns_str})
                SELECT {columns_str}
                FROM social_search_2
                ORDER BY {time_column} DESC
                LIMIT 700000;
            """

            print("Executing data copy... This may take a while.")
            await conn.execute(copy_sql)

            # Verify the copy
            final_count = await conn.fetchval(
                "SELECT COUNT(*) FROM social_search_prefs"
            )
            print(f"Data copy completed! Records copied: {final_count}")
        else:
            print(f"Table already has {current_count} records. Skipping data copy.")

        # Generate embeddings for posts that don't have them yet
        print("Checking for posts without embeddings...")
        unembedded_count = await conn.fetchval("""
            SELECT COUNT(*) FROM social_search_prefs
            WHERE qwen_vector IS NULL
        """)

        if unembedded_count > 0:
            print(
                f"Found {unembedded_count} posts without embeddings. Generating vectors..."
            )

            # Load the model
            model = load_qwen_model()

            # Process in batches to avoid memory issues
            batch_size = 100
            processed = 0

            # Get posts that need embeddings, assuming common column names
            # We'll check what columns are actually available
            column_check = await conn.fetch("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'social_search_prefs'
                AND column_name IN ('title', 'description', 'content', 'text', 'body')
                ORDER BY
                    CASE column_name
                        WHEN 'title' THEN 1
                        WHEN 'description' THEN 2
                        WHEN 'content' THEN 3
                        WHEN 'text' THEN 4
                        WHEN 'body' THEN 5
                        ELSE 6
                    END
            """)

            available_columns = [row["column_name"] for row in column_check]
            print(f"Available text columns: {available_columns}")

            if not available_columns:
                print(
                    "Warning: No title, description, content, text, or body columns found!"
                )
                # Let's see all available columns
                all_columns = await conn.fetch("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'social_search_prefs'
                    ORDER BY ordinal_position
                """)
                print("All available columns:")
                for col in all_columns:
                    print(f"  - {col['column_name']}")
                return

            # Use the first two available columns (or just one if only one exists)
            title_col = available_columns[0] if len(available_columns) > 0 else None
            desc_col = available_columns[1] if len(available_columns) > 1 else None

            print(f"Using columns: title='{title_col}', description='{desc_col}'")

            # Build the SELECT query
            select_cols = ["id"]  # Assuming there's an id column
            if title_col:
                select_cols.append(title_col)
            if desc_col:
                select_cols.append(desc_col)

            while processed < unembedded_count:
                # Get a batch of unembedded posts
                query = f"""
                    SELECT {", ".join(select_cols)}
                    FROM social_search_prefs
                    WHERE qwen_vector IS NULL
                    ORDER BY id
                    LIMIT {batch_size} OFFSET {processed}
                """

                batch = await conn.fetch(query)
                if not batch:
                    break

                print(
                    f"Processing batch {processed // batch_size + 1} ({len(batch)} posts)..."
                )

                # Generate embeddings for this batch
                for row in batch:
                    title = row.get(title_col) if title_col else ""
                    description = row.get(desc_col) if desc_col else ""

                    embedding = generate_embedding(model, title, description)

                    if embedding:
                        # Update the post with its embedding
                        await conn.execute(
                            """
                            UPDATE social_search_prefs
                            SET qwen_vector = $1::vector
                            WHERE id = $2
                        """,
                            embedding,
                            row["id"],
                        )

                processed += len(batch)
                print(
                    f"Progress: {processed}/{unembedded_count} posts processed ({processed / unembedded_count * 100:.1f}%)"
                )

            print("Embedding generation completed!")
        else:
            print("All posts already have embeddings.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
