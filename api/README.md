Run and test the BloomWatch API (local)

1) Create an Anaconda environment (recommended)

   conda create -n bloombly python=3.11 -y
   conda activate bloombly

2) Install dependencies via conda-forge (avoids compiling pandas from source on macOS M1/ARM):

   conda install -c conda-forge earthengine-api geemap pandas numpy flask flask-cors google-auth python-dotenv -y

3) Configure environment variables

   - Copy `.env.example` to `.env` and fill in `EE_PROJECT` and `GOOGLE_APPLICATION_CREDENTIALS_JSON`.
   - Alternatively set them in your shell:

```bash
export EE_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type":... }'
```

4) Run the API

```bash
python app/main.py
```

5) Test the blooms endpoint

```bash
curl "http://localhost:5000/api/blooms?aoi_type=global&start_date=2024-05-01&end_date=2024-07-01&time_step_days=15"
```

Notes and troubleshooting
- If you see an `invalid_scope` or authentication errors, ensure your service account key is a JSON service account key (not OAuth client ID) and that the key has the proper IAM roles: `Earth Engine Service Agent` or project-level roles to access Earth Engine. Use the `google-auth` scopes:
  - https://www.googleapis.com/auth/earthengine
  - https://www.googleapis.com/auth/cloud-platform

- For large exports you should use `ee.batch.Export` to Cloud Storage and not rely on `getInfo()`.
