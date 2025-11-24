import sys
import base64
import requests
import json
from pathlib import Path


def analyze_document(file_path: str, questions: list = None, server_url: str = "http://127.0.0.1:3000"):
    """Send a document to the API for analysis."""
    path = Path(file_path)
    
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        return
    
    with open(path, 'rb') as f:
        file_data = f.read()
    
    encoded_data = base64.b64encode(file_data).decode('utf-8')

    request_data = {
        'data': encoded_data,
        'filename': path.name
    }
    
    if questions:
        request_data['questions'] = questions
    
    print(f"Analyzing document: {path.name}")
    print(f"File size: {len(file_data)} bytes")
    if questions:
        print(f"Questions: {len(questions)}")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q}")
    print(f"Server URL: {server_url}/api/v1/analyze")
    print()
    
    try:
        response = requests.post(
            f"{server_url}/api/v1/analyze",
            json=request_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Response Status: {response.status_code}")
        print()
        
        if response.status_code == 200:
            result = response.json()
            script_dir = Path(__file__).parent
            result_file = script_dir / "result.json"
            
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"Results written to: {result_file}")
        else:
            error_data = response.json()
            print("Error Response:")
            print(json.dumps(error_data, indent=2))
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server.")
        print("Make sure the server is running with: cargo run server")
    except Exception as e:
        print(f"Error: {e}")


def check_health(server_url: str = "http://127.0.0.1:3000"):
    """Check if the server is healthy."""
    try:
        response = requests.get(f"{server_url}/health")
        
        if response.status_code == 200:
            health_data = response.json()
            print("Server Health Check:")
            print(json.dumps(health_data, indent=2))
            return True
        else:
            print(f"Server returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server.")
        print("Make sure the server is running with: cargo run server")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <file_path> [question1] [question2] ...")
        print("       python client.py health")
        print()
        print("Examples:")
        print("  python examples/client.py tests/fixtures/png/test.png")
        print("  python examples/client.py tests/fixtures/pdf/test.pdf")
        print("  python examples/client.py tests/fixtures/pdf/test.pdf 'What is the document about?'")
        print("  python examples/client.py tests/fixtures/pdf/test.pdf 'What is the title?' 'Who is the author?'")
        print("  python examples/client.py health")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "health":
        check_health()
    else:
        file_path = command
        questions = sys.argv[2:] if len(sys.argv) > 2 else None
        analyze_document(file_path, questions)


if __name__ == "__main__":
    main()
