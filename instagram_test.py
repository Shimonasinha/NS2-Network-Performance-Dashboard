import instaloader

# Create instance
L = instaloader.Instaloader()

print("🔍 Testing Instagram scraping (no login)...\n")

# Test on a public fitness account
test_username = "kayla_itsines"  # Popular fitness influencer

try:
    # Get profile
    profile = instaloader.Profile.from_username(L.context, test_username)
    
    print(f"✅ Successfully accessed profile!\n")
    print(f"Username: {profile.username}")
    print(f"Full Name: {profile.full_name}")
    print(f"Followers: {profile.followers:,}")
    print(f"Following: {profile.followees:,}")
    print(f"Posts: {profile.mediacount}")
    print(f"Bio: {profile.biography[:100]}...")
    print(f"Private: {profile.is_private}")
    
    print(f"\n📸 Getting recent posts...\n")
    
    # Get 3 recent posts
    post_count = 0
    for post in profile.get_posts():
        if post_count >= 3:
            break
        
        print(f"Post {post_count + 1}:")
        print(f"  Date: {post.date}")
        print(f"  Likes: {post.likes:,}")
        print(f"  Comments: {post.comments}")
        print(f"  Caption: {post.caption[:80] if post.caption else 'No caption'}...")
        print()
        
        post_count += 1
    
    print("✅ Test successful! Instaloader is working!")

except Exception as e:
    print(f"❌ Error: {e}")
```
```
Save as: instagram_test.py
Location: D:\CAPSTONE\YoutubeScrapper\