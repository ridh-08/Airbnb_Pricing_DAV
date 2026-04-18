 # Airbnb Pricing and Neighbourhood Dynamics (CSE520)

 This repository contains the complete multi-city analytics project for Airbnb pricing and neighbourhood dynamics.

 ## Team

 - Riddhi Bhargava (AU2220028)
 - Vaishnavi Dahihande (AU2220080)

 ## Cities Covered

 - New York
 - Chicago
 - Austin
 - Los Angeles
 - Nashville
 - New Orleans

 ## Project Structure

 - `src/` core analysis modules (cleaning, feature engineering, modeling, clustering, spatial, calendar, statistics)
 - `data/processed/` cleaned and engineered datasets used by analysis modules
 - `outputs/` generated tables, plots, map files, and consolidated reports
 - `templates/` dashboard template
 - `app.py` Flask backend and dashboard data APIs
 - `export_static_dashboard.py` static dashboard exporter
 - `run_pipeline.py` end-to-end pipeline runner

 ## Main Outputs

 The project produces:

 - city-level tables and plots in `outputs/<city>/tables` and `outputs/<city>/plots`
 - cross-city outputs in `outputs/multi_city/tables`
 - consolidated workbook in `outputs/consolidated/airbnb_multicity_summary.xlsx`
 - static dashboard bundle in `netlify_static/`

 ## Run Instructions

 ### 1) Full pipeline

 ```powershell
 C:/Users/Riddhi/AppData/Local/Programs/Python/Python311/python.exe run_pipeline.py
 ```

 ### 2) Launch dashboard (Flask)

 ```powershell
 C:/Users/Riddhi/AppData/Local/Programs/Python/Python311/python.exe app.py
 ```

 Open: `http://127.0.0.1:5050`

 ### 3) Export static dashboard

 ```powershell
 C:/Users/Riddhi/AppData/Local/Programs/Python/Python311/python.exe export_static_dashboard.py
 ```

 Static output:

 - `netlify_static/index.html`
 - `netlify_static/maps/*.html`

 ## Deployment

 ### Render (backend)

 Configured using `render.yaml`:

 - Build: `pip install -r requirements.txt`
 - Start: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 180`

 ### Netlify (static dashboard)

 Deploy `netlify_static/` as the publish folder.

 If using backend APIs from static hosting, set `API_BASE_URL` before exporting:

 ```powershell
 set API_BASE_URL=https://<your-render-service>.onrender.com
 C:/Users/Riddhi/AppData/Local/Programs/Python/Python311/python.exe export_static_dashboard.py
 ```

 ## Final Submission Notes

 - Core code for evaluation is under `src/`, `app.py`, `templates/`, and runner scripts.
 - Generated analytical evidence is under `outputs/`.
 - The dashboard supports both Flask-served mode and static-export mode.
