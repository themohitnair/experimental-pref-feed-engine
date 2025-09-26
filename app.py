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


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # Handle zero vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


async def get_db_connection():
    return await asyncpg.connect(**db_config)


async def load_user_vectors():
    """Load user vectors from database into memory"""
    conn = await get_db_connection()
    try:
        users = await conn.fetch("SELECT username, user_vector FROM user_prefs_api")
        for user in users:
            # Convert PostgreSQL vector to Python list
            vector_str = user["user_vector"]
            # Parse vector string [1.0,2.0,3.0,...] to list
            vector_list = eval(vector_str)  # In production, use proper parsing
            user_vectors[user["username"]] = np.array(vector_list)
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
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1400px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 30px; }
            .feeds-container { display: flex; gap: 20px; }
            .user-feed { flex: 1; background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .feed-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 2px solid #eee; padding-bottom: 10px; }
            .refresh-btn { background: #28a745; color: white; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer; }
            .refresh-btn:hover { background: #218838; }
            .post { margin-bottom: 15px; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #007bff; }
            .post-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
            .similarity-score { background: #e9ecef; padding: 4px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold; }
            .post-content h4 { margin: 0 0 8px 0; color: #333; }
            .post-content p { margin: 0 0 10px 0; color: #666; line-height: 1.4; }
            .post-actions { display: flex; gap: 10px; align-items: center; }
            .like-btn { border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 0.9em; }
            .like-btn.liked { background: #dc3545; color: white; }
            .like-btn.not-liked { background: #007bff; color: white; }
            .like-btn:hover:not(:disabled) { opacity: 0.8; }
            .like-btn:disabled { opacity: 0.6; cursor: not-allowed; }
            .status { position: fixed; top: 20px; right: 20px; padding: 10px 15px; border-radius: 5px; z-index: 1000; }
            .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .loading { text-align: center; padding: 20px; color: #666; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 Preference Feed Engine</h1>
                <p>AI-powered personalized feeds with semantic similarity</p>
            </div>

            <div class="feeds-container">
                <div class="user-feed">
                    <div class="feed-header">
                        <h2>👤 User1 Feed</h2>
                        <button class="refresh-btn" onclick="refreshFeed('user1')">🔄 Refresh</button>
                    </div>
                    <div id="user1-posts" class="loading">Loading posts...</div>
                </div>

                <div class="user-feed">
                    <div class="feed-header">
                        <h2>👤 User2 Feed</h2>
                        <button class="refresh-btn" onclick="refreshFeed('user2')">🔄 Refresh</button>
                    </div>
                    <div id="user2-posts" class="loading">Loading posts...</div>
                </div>
            </div>

            <div id="status"></div>
        </div>

        <script>
            let user1Posts = [];
            let user2Posts = [];

            async function loadInitialPosts() {
                try {
                    const response = await fetch('/posts');
                    const posts = await response.json();
                    user1Posts = posts;
                    user2Posts = posts;
                    await renderFeeds();
                } catch (error) {
                    showStatus('Error loading posts: ' + error.message, true);
                }
            }

            async function refreshFeed(username) {
                try {
                    showStatus(`Refreshing ${username} feed...`);
                    const response = await fetch(`/feed/${username}`);
                    const posts = await response.json();

                    if (username === 'user1') {
                        user1Posts = posts;
                        await renderFeed('user1', user1Posts);
                    } else {
                        user2Posts = posts;
                        await renderFeed('user2', user2Posts);
                    }

                    showStatus(`${username} feed refreshed with personalized recommendations!`);
                } catch (error) {
                    showStatus('Error refreshing feed: ' + error.message, true);
                }
            }

            async function renderFeeds() {
                await renderFeed('user1', user1Posts);
                await renderFeed('user2', user2Posts);
            }

            async function renderFeed(username, posts) {
                const container = document.getElementById(`${username}-posts`);
                container.innerHTML = '';

                for (const post of posts) {
                    const postDiv = await createPostDiv(post, username);
                    container.appendChild(postDiv);
                }
            }

            async function createPostDiv(post, username) {
                const div = document.createElement('div');
                div.className = 'post';

                // Check if post is liked
                let isLiked = false;
                try {
                    const response = await fetch(`/is-liked/${username}/${post.id}`);
                    const result = await response.json();
                    isLiked = result.is_liked;
                } catch (error) {
                    console.log('Error checking like status:', error);
                }

                const similarityColor = post.similarity_score > 0.5 ? '#28a745' : post.similarity_score > 0.2 ? '#ffc107' : '#6c757d';

                div.innerHTML = `
                    <div class="post-header">
                        <span class="similarity-score" style="background-color: ${similarityColor}; color: white;">
                            Similarity: ${(post.similarity_score * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div class="post-content">
                        <h4>${post.title || 'No Title'}</h4>
                        <p>${post.description || 'No Description'}</p>
                    </div>
                    <div class="post-actions">
                        <button class="like-btn ${isLiked ? 'liked' : 'not-liked'}"
                                onclick="${isLiked ? 'unlikePost' : 'likePost'}('${username}', ${post.id}, this)">
                            ${isLiked ? '💔 Unlike' : '👍 Like'}
                        </button>
                    </div>
                `;
                return div;
            }

            async function likePost(username, postId, buttonElement) {
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
                        // Update button state
                        buttonElement.className = 'like-btn liked';
                        buttonElement.innerHTML = '💔 Unlike';
                        buttonElement.onclick = () => unlikePost(username, postId, buttonElement);

                        showStatus(`${username} liked post! Vector updated with preferences.`);
                    } else {
                        showStatus('Error: ' + result.detail, true);
                    }
                } catch (error) {
                    showStatus('Error liking post: ' + error.message, true);
                }
            }

            async function unlikePost(username, postId, buttonElement) {
                try {
                    const response = await fetch('/unlike', {
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
                        // Update button state
                        buttonElement.className = 'like-btn not-liked';
                        buttonElement.innerHTML = '👍 Like';
                        buttonElement.onclick = () => likePost(username, postId, buttonElement);

                        showStatus(`${username} unliked post! Vector preferences reversed.`);
                    } else {
                        showStatus('Error: ' + result.detail, true);
                    }
                } catch (error) {
                    showStatus('Error unliking post: ' + error.message, true);
                }
            }

            function showStatus(message, isError = false) {
                const status = document.getElementById('status');
                status.innerHTML = `<div class="status ${isError ? 'error' : 'success'}">${message}</div>`;
                setTimeout(() => {
                    status.innerHTML = '';
                }, 3000);
            }

            // Load posts on page load
            loadInitialPosts();
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
            LIMIT 15
        """)

        # Add zero similarity score for initial random posts
        result = []
        for post in posts:
            post_dict = dict(post)
            post_dict['similarity_score'] = 0.0
            result.append(post_dict)

        return result
    finally:
        await conn.close()

@app.get("/feed/{username}")
async def get_personalized_feed(username: str):
    """Get personalized feed based on user's vector similarity"""
    if username not in user_vectors:
        raise HTTPException(status_code=404, detail="User not found")

    conn = await get_db_connection()
    try:
        # Get all posts with vectors
        posts = await conn.fetch("""
            SELECT id, title, description, qwen_vector
            FROM social_search_prefs
            WHERE qwen_vector IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 100
        """)

        user_vector = user_vectors[username]
        scored_posts = []

        for post in posts:
            post_vector_str = post['qwen_vector']
            post_vector = np.array(eval(post_vector_str))

            similarity = cosine_similarity(user_vector, post_vector)

            scored_posts.append({
                'id': post['id'],
                'title': post['title'],
                'description': post['description'],
                'similarity_score': similarity
            })

        # Sort by similarity score (highest first)
        scored_posts.sort(key=lambda x: x['similarity_score'], reverse=True)

        return scored_posts[:15]  # Return top 15
    finally:
        await conn.close()

@app.get("/is-liked/{username}/{post_id}")
async def check_if_liked(username: str, post_id: int):
    """Check if a user has liked a specific post"""
    conn = await get_db_connection()
    try:
        user_id = await conn.fetchval(
            "SELECT id FROM user_prefs_api WHERE username = $1", username
        )

        if not user_id:
            raise HTTPException(status_code=404, detail="User not found")

        liked = await conn.fetchval("""
            SELECT EXISTS(
                SELECT 1 FROM user_likes
                WHERE user_id = $1 AND post_id = $2
            )
        """, user_id, post_id)

        return {"is_liked": liked}
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
        post_data = await conn.fetchrow(
            """
            SELECT qwen_vector FROM social_search_prefs WHERE id = $1
        """,
            request.post_id,
        )

        if not post_data:
            raise HTTPException(status_code=404, detail="Post not found")

        # Parse post vector
        post_vector_str = post_data["qwen_vector"]
        post_vector = np.array(
            eval(post_vector_str)
        )  # In production, use proper parsing

        # Get current user vector
        current_user_vector = user_vectors[request.username]

        # Exponential moving average (learning rate approach)
        alpha = 0.15  # Learning rate - gives more weight to recent interactions
        updated_vector = alpha * post_vector + (1 - alpha) * current_user_vector
        user_vectors[request.username] = updated_vector

        # Update database
        updated_vector_str = "[" + ",".join(map(str, updated_vector.tolist())) + "]"
        await conn.execute(
            """
            UPDATE user_prefs_api
            SET user_vector = $1
            WHERE username = $2
        """,
            updated_vector_str,
            request.username,
        )

        # Record the like
        user_id = await conn.fetchval(
            "SELECT id FROM user_prefs_api WHERE username = $1", request.username
        )
        await conn.execute(
            """
            INSERT INTO user_likes (user_id, post_id)
            VALUES ($1, $2)
            ON CONFLICT (user_id, post_id) DO NOTHING
        """,
            user_id,
            request.post_id,
        )

        return {
            "message": f"User {request.username} liked post {request.post_id}. Vector updated!"
        }

    finally:
        await conn.close()

@app.post("/unlike")
async def unlike_post(request: LikeRequest):
    """Handle user unliking a post - reverses the vector operation"""
    if request.username not in user_vectors:
        raise HTTPException(status_code=404, detail="User not found")

    conn = await get_db_connection()
    try:
        # Check if the user actually liked this post
        user_id = await conn.fetchval(
            "SELECT id FROM user_prefs_api WHERE username = $1", request.username
        )

        liked = await conn.fetchval("""
            SELECT EXISTS(
                SELECT 1 FROM user_likes
                WHERE user_id = $1 AND post_id = $2
            )
        """, user_id, request.post_id)

        if not liked:
            raise HTTPException(status_code=400, detail="Post not liked by user")

        # Get the post vector
        post_data = await conn.fetchrow(
            "SELECT qwen_vector FROM social_search_prefs WHERE id = $1",
            request.post_id,
        )

        if not post_data:
            raise HTTPException(status_code=404, detail="Post not found")

        # Parse post vector
        post_vector_str = post_data["qwen_vector"]
        post_vector = np.array(eval(post_vector_str))

        # Get current user vector
        current_user_vector = user_vectors[request.username]

        # Reverse the exponential moving average: if new_vec = α * post_vec + (1-α) * old_vec
        # then old_vec = (new_vec - α * post_vec) / (1-α)
        alpha = 0.15  # Same learning rate as in like operation
        updated_vector = (current_user_vector - alpha * post_vector) / (1 - alpha)
        user_vectors[request.username] = updated_vector

        # Update database
        updated_vector_str = "[" + ",".join(map(str, updated_vector.tolist())) + "]"
        await conn.execute(
            "UPDATE user_prefs_api SET user_vector = $1 WHERE username = $2",
            updated_vector_str,
            request.username,
        )

        # Remove the like record
        await conn.execute(
            "DELETE FROM user_likes WHERE user_id = $1 AND post_id = $2",
            user_id,
            request.post_id,
        )

        return {
            "message": f"User {request.username} unliked post {request.post_id}. Vector updated!"
        }

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
        "vector_norm": float(np.linalg.norm(vector)),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6962)
