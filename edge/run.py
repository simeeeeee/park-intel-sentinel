import time
import requests

def wait_for_backend(url, retries=10, delay=3):
    for i in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("✅ Backend is ready.")
                return True
        except Exception as e:
            print(f"🔁 Waiting for backend... ({i+1}/{retries})")
        time.sleep(delay)
    print("❌ Backend not ready. Exiting.")
    exit(1)

# 사용 예시
if __name__ == "__main__":
    wait_for_backend("http://backend:8000/health")
