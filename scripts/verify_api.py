import requests

try:
    res = requests.get("http://127.0.0.1:5000/api/xai/1")
    if res.status_code == 200:
        data = res.json()
        print("✅ API XAI returned 200")
        if "saliency" in data:
            sal = data["saliency"]
            print(f"✅ Saliency found, length: {len(sal)}")
            if len(sal) > 0:
                print(f"   First value: {sal[0]}")
        else:
            print("❌ Saliency NOT found in response")
            
        if "explanation" in data:
            print(f"✅ Explanation: {data['explanation'][:50]}...")
        else:
            print("❌ Explanation NOT found")
            
        if "classes" in data:
            print(f"✅ Classes found: {len(data['classes'])}")
        else:
            print("❌ Classes NOT found")
    else:
        print(f"❌ API XAI failed with {res.status_code}")
        print(res.text)
except Exception as e:
    print(f"❌ Connection failed: {e}")
