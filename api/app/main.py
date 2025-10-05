from flask import Flask, jsonify
from flask_cors import CORS
import ee
import os
import logging
from google.oauth2 import service_account
import config
from routes.data import data_bp
from routes.predict import predict_bp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

def initialize_earth_engine():
    """Initializes the Earth Engine API with service account credentials."""
    try:
        credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON', config.GOOGLE_APPLICATION_CREDENTIALS_JSON)
        project_id = os.getenv('EE_PROJECT', config.EE_PROJECT)

        if credentials_json and project_id:
            credentials_dict = json.loads(credentials_json)
            SCOPES = [
                'https://www.googleapis.com/auth/earthengine',
                'https://www.googleapis.com/auth/cloud-platform',
                'https://www.googleapis.com/auth/userinfo.email'
            ]
            credentials = service_account.Credentials.from_service_account_info(
                credentials_dict,
                scopes=SCOPES
            )
            ee.Initialize(credentials=credentials, project=project_id)
            logging.info("Earth Engine initialized successfully with service account.")
        else:
            ee.Initialize()
            logging.info("Earth Engine initialized with default credentials.")

    except Exception as e:
        logging.error(f"Error initializing Earth Engine: {e}")
        logging.error("Please ensure EE_PROJECT and GOOGLE_APPLICATION_CREDENTIALS_JSON are set.")
        raise

initialize_earth_engine()

app.register_blueprint(data_bp, url_prefix='/api')
app.register_blueprint(predict_bp, url_prefix='/api')

@app.route('/')
def index():
    """Returns a welcome message."""
    return jsonify(message="BloomWatch API is running.")

if __name__ == '__main__':
    port = int(os.getenv('PORT', config.PORT))
    app.run(host='0.0.0.0', port=port, debug=config.DEBUG)
