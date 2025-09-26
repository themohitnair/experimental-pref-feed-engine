from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import asyncpg
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

app = FastAPI(title="Preference Feed Engine")

# In-memory storage for user vectors
user_vectors = {}

# Database connection config
db_config = {
    "user": os.getenv("PSQL_DB_USERNAME"),
    "password": os.getenv("PSQL_DB_PWD"),
    "host": os.getenv("PSQL_DB_HOSTNAME"),
    "database": os.getenv("PSQL_DB"),
    "port": int(os.getenv("PSQL_DB_PORT", 5432)),
}

class LikeRequest(BaseModel):
    username: str
    post_id: int

async def get_db_connection():
    return await asyncpg.connect(**db_config)

async def load_user_vectors():
    """Load user vectors from database into memory"""
    conn = await get_db_connection()
    try:
        users = await conn.fetch("SELECT username, user_vector FROM user_prefs_api")
        for user in users:
            # Convert PostgreSQL vector to Python list
            vector_str = user['user_vector']
            # Parse vector string [1.0,2.0,3.0,...] to list
            vector_list = eval(vector_str)  # In production, use proper parsing
            user_vectors[user['username']] = np.array(vector_list)
        print(f"Loaded {len(user_vectors)} user vectors into memory")
    finally:
        await conn.close()

@app.on_event("startup")
async def startup_event():
    await load_user_vectors()

@app.get("/", response_class=HTMLResponse)
async def get_html():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Preference Feed Engine</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .user-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            .post { margin: 10px 0; padding: 10px; background: #f9f9f9; border-radius: 3px; }
            .like-btn { background: #007bff; color: white; border: none; padding: 5px 15px; border-radius: 3px; cursor: pointer; }
            .like-btn:hover { background: #0056b3; }
            .status { margin: 10px 0; padding: 10px; background: #e9f7ef; border-radius: 3px; }
            .error { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Preference Feed Engine</h1>

            <div class="user-section">
                <h2>User1 Feed</h2>
                <div id="user1-posts"></div>
            </div>

            <div class="user-section">
                <h2>User2 Feed</h2>
                <div id="user2-posts"></div>
            </div>

            <div id="status"></div>
        </div>

        <script>
            let posts = [];

            async function loadPosts() {
                try {
                    const response = await fetch('/posts');
                    posts = await response.json();
                    renderPosts();
                } catch (error) {
                    showStatus('Error loading posts: ' + error.message, true);
                }
            }

            function renderPosts() {
                const user1Posts = document.getElementById('user1-posts');
                const user2Posts = document.getElementById('user2-posts');

                user1Posts.innerHTML = '';
                user2Posts.innerHTML = '';

                posts.slice(0, 10).forEach(post => {
                    const postDiv1 = createPostDiv(post, 'user1');
                    const postDiv2 = createPostDiv(post, 'user2');
                    user1Posts.appendChild(postDiv1);
                    user2Posts.appendChild(postDiv2);
                });
            }

            function createPostDiv(post, username) {
                const div = document.createElement('div');
                div.className = 'post';
                div.innerHTML = `
                    <h4>${post.title || 'No Title'}</h4>
                    <p>${post.description || 'No Description'}</p>
                    <button class="like-btn" onclick="likePost('${username}', ${post.id})">
                        Like
                    </button>
                `;
                return div;
            }

            async function likePost(username, postId) {
                try {
                    const response = await fetch('/like', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            username: username,
                            post_id: postId
                        })
                    });

                    const result = await response.json();
                    if (response.ok) {
                        showStatus(`${username} liked post ${postId}. Vector updated!`);
                    } else {
                        showStatus('Error: ' + result.detail, true);
                    }
                } catch (error) {
                    showStatus('Error liking post: ' + error.message, true);
                }
            }

            function showStatus(message, isError = false) {
                const status = document.getElementById('status');
                status.innerHTML = `<div class="status ${isError ? 'error' : ''}">${message}</div>`;
                setTimeout(() => {
                    status.innerHTML = '';
                }, 3000);
            }

            // Load posts on page load
            loadPosts();
        </script>
    </body>
    </html>
    """

@app.get("/posts")
async def get_posts():
    """Get a sample of posts for the frontend"""
    conn = await get_db_connection()
    try:
        posts = await conn.fetch("""
            SELECT id, title, description
            FROM social_search_prefs
            WHERE qwen_vector IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 20
        """)
        return [dict(post) for post in posts]
    finally:
        await conn.close()

@app.post("/like")
async def like_post(request: LikeRequest):
    """Handle user liking a post - updates user vector"""
    if request.username not in user_vectors:
        raise HTTPException(status_code=404, detail="User not found")

    conn = await get_db_connection()
    try:
        # Get the post vector
        post_data = await conn.fetchrow("""
            SELECT qwen_vector FROM social_search_prefs WHERE id = $1
        """, request.post_id)

        if not post_data:
            raise HTTPException(status_code=404, detail="Post not found")

        # Parse post vector
        post_vector_str = post_data['qwen_vector']
        post_vector = np.array(eval(post_vector_str))  # In production, use proper parsing

        # Get current user vector
        current_user_vector = user_vectors[request.username]

        # Average the vectors (simple preference update)
        updated_vector = (current_user_vector + post_vector) / 2
        user_vectors[request.username] = updated_vector

        # Update database
        updated_vector_str = '[' + ','.join(map(str, updated_vector.tolist())) + ']'
        await conn.execute("""
            UPDATE user_prefs_api
            SET user_vector = $1
            WHERE username = $2
        """, updated_vector_str, request.username)

        # Record the like
        user_id = await conn.fetchval("SELECT id FROM user_prefs_api WHERE username = $1", request.username)
        await conn.execute("""
            INSERT INTO user_likes (user_id, post_id)
            VALUES ($1, $2)
            ON CONFLICT (user_id, post_id) DO NOTHING
        """, user_id, request.post_id)

        return {"message": f"User {request.username} liked post {request.post_id}. Vector updated!"}

    finally:
        await conn.close()

@app.get("/user/{username}/vector")
async def get_user_vector(username: str):
    """Get current user vector (first 10 dimensions for display)"""
    if username not in user_vectors:
        raise HTTPException(status_code=404, detail="User not found")

    vector = user_vectors[username]
    return {
        "username": username,
        "vector_preview": vector[:10].tolist(),
        "vector_norm": float(np.linalg.norm(vector))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)