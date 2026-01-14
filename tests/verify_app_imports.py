try:
    from app import pipeline_google
    print("app.pipeline_google imported successfully")
except Exception as e:
    print(f"Failed to import app.pipeline_google: {e}")
    exit(1)

try:
    from app import filters
    print("app.filters imported successfully")
except Exception as e:
    print(f"Failed to import app.filters: {e}")
    exit(1)

try:
    from app import pipeline_groq
    print("app.pipeline_groq imported successfully")
except Exception as e:
    print(f"Failed to import app.pipeline_groq: {e}")
    # Don't exit, might fail due to missing optional deps, but we check syntax/names
    pass 

try:
    from app import pipeline_ollama
    print("app.pipeline_ollama imported successfully")
except Exception as e:
    print(f"Failed to import app.pipeline_ollama: {e}")
    pass
