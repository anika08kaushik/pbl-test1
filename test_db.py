from database import init_db, create_user, get_jobs
try:
    print("Initializing database...")
    init_db()
    print("Database initialized.")
    
    print("Creating demo user...")
    try:
        user = create_user("demo@smarthire.ai", "password123", "RECRUITER")
        print(f"User created: {user.id}")
    except Exception as e:
        print(f"User creation failed (maybe already exists): {e}")
        
    print("Fetching jobs for user 1...")
    jobs = get_jobs(1)
    print(f"Jobs found: {len(jobs)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
