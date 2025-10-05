from flask import Flask, jsonify
from flask_cors import CORS
import os
import logging
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from . import config
    from .routes.data import data_bp
    from .routes.predict import predict_bp
    from .routes.sakura import sakura_bp
except ImportError:
    import config
    from routes.data import data_bp
    from routes.predict import predict_bp
    from routes.sakura import sakura_bp

# Try to import Earth Engine, but make it optional
try:
    import ee
    from google.oauth2 import service_account
    EE_MODULE_AVAILABLE = True
except ImportError:
    EE_MODULE_AVAILABLE = False
    logging.warning("Earth Engine module not installed - will use fallback data")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# Global flag to track if Earth Engine is available
EARTH_ENGINE_AVAILABLE = False

def initialize_earth_engine():
    """
    Initializes the Earth Engine API with service account credentials.
    Returns True if successful, False otherwise.
    Earth Engine is OPTIONAL - the API works without it using fallback climate data.
    """
    global EARTH_ENGINE_AVAILABLE
    
    if not EE_MODULE_AVAILABLE:
        logging.warning("⚠ Earth Engine module not available")
        logging.info("⚠ Continuing without Earth Engine - will use fallback climate data")
        return False
    
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
            logging.info("✓ Earth Engine initialized successfully with service account.")
            EARTH_ENGINE_AVAILABLE = True
            return True
        else:
            ee.Initialize()
            logging.info("✓ Earth Engine initialized with default credentials.")
            EARTH_ENGINE_AVAILABLE = True
            return True

    except Exception as e:
        logging.warning(f"⚠ Earth Engine initialization failed: {e}")
        logging.info("⚠ Continuing without Earth Engine - will use fallback climate data")
        logging.info("  To enable Earth Engine, set up credentials in .env file:")
        logging.info("    EE_PROJECT=your-project-id")
        logging.info("    GOOGLE_APPLICATION_CREDENTIALS_JSON='{...}'")
        EARTH_ENGINE_AVAILABLE = False
        return False

# Try to initialize Earth Engine, but don't fail if it's not available
initialize_earth_engine()

app.register_blueprint(data_bp, url_prefix='/api/data')
app.register_blueprint(predict_bp, url_prefix='/api/predict')
app.register_blueprint(sakura_bp, url_prefix='/api/sakura')

@app.route('/')
def index():
    """Returns a welcome message."""
    return jsonify(message="BloomWatch API is running.")

@app.route('/api/health')
def health():
    """Returns API health status."""
    return jsonify({
        "status": "healthy",
        "earth_engine_available": EARTH_ENGINE_AVAILABLE,
        "message": "API is running" + (" with Earth Engine" if EARTH_ENGINE_AVAILABLE else " (fallback mode)")
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', config.PORT))
    app.run(host='0.0.0.0', port=port, debug=config.DEBUG)
