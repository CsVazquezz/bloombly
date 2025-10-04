from flask import Flask, jsonify
from flask_cors import CORS
import ee
import json
import os
from google.oauth2 import service_account
from dotenv import load_dotenv
from routes.data import data_bp
from routes.predict import predict_bp

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Earth Engine with service account
def initialize_earth_engine():
    try:
        # Get credentials from environment variable
        credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
        project_id = os.getenv('EE_PROJECT')

        if credentials_json and project_id:
            # Parse the JSON credentials
            import json
            credentials_dict = json.loads(credentials_json)

            # Required scopes for Earth Engine + Cloud APIs. Including cloud-platform
            # helps when the Earth Engine client needs to access ancillary Google APIs.
            SCOPES = [
                'https://www.googleapis.com/auth/earthengine',
                'https://www.googleapis.com/auth/cloud-platform',
                'https://www.googleapis.com/auth/userinfo.email'
            ]

            # Build credentials with explicit scopes. This prevents the
            # `invalid_scope` / `ID token audience` error when the key lacks
            # an OAuth scope.
            credentials = service_account.Credentials.from_service_account_info(
                credentials_dict,
                scopes=SCOPES
            )

            # Initialize with project and credentials
            ee.Initialize(credentials=credentials, project=project_id)
            print("Earth Engine initialized successfully with service account")
        else:
            # Fallback to default authentication (for development)
            ee.Initialize()
            print("Earth Engine initialized with default credentials")

    except Exception as e:
        print(f"Earth Engine initialization failed: {e}")
        print("Make sure EE_PROJECT and GOOGLE_APPLICATION_CREDENTIALS_JSON environment variables are set")
        raise e

# Initialize on startup
initialize_earth_engine()

app.register_blueprint(data_bp, url_prefix='/api')
app.register_blueprint(predict_bp, url_prefix='/api')

@app.route('/')
def hello():
    return jsonify({"message": "BloomWatch API"})

if __name__ == '__main__':
    # Allow overriding port via environment variable (helpful when 5000 is occupied)
    try:
        port = int(os.getenv('PORT', '5000'))
    except ValueError:
        port = 5000

    try:
        app.run(debug=True, host='0.0.0.0', port=port)
    except OSError as e:
        if 'Address already in use' in str(e):
            print(f"Port {port} is already in use. Start with a different PORT, e.g. PORT=5001 python app/main.py")
        raise