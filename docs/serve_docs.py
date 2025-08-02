#!/usr/bin/env python3
"""
Simple script to serve QEM documentation locally
"""
import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

def serve_docs(port=8000):
    """Serve documentation on local server"""
    
    # Change to documentation directory
    docs_dir = Path(__file__).parent / "build" / "html"
    
    if not docs_dir.exists():
        print("❌ Documentation not built. Run 'make html' first.")
        sys.exit(1)
    
    os.chdir(docs_dir)
    
    # Create server
    handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"🚀 Serving QEM documentation at http://localhost:{port}")
            print(f"📁 Documentation directory: {docs_dir}")
            print("Press Ctrl+C to stop the server")
            
            # Open browser
            webbrowser.open(f"http://localhost:{port}")
            
            # Serve forever
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n👋 Documentation server stopped")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {port} is already in use. Try a different port:")
            print(f"python serve_docs.py {port + 1}")
        else:
            print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    serve_docs(port)