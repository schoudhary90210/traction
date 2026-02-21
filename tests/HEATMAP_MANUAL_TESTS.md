# Heatmap Manual Smoke Tests

## Start the app

```bash
cd /Users/siddhant/Downloads/Agri-Scout
./venv/bin/streamlit run app.py
```

Open `http://localhost:8501`.

## Core verification flow

1. Confirm `Ghost Tractor — Field Map` appears with no error banner.
2. In sidebar, click `🚨 Inject 10 Fake Detections`.
3. Click `🔄 Refresh Map & Telemetry`.
4. Verify:
   - detection count increases,
   - red disease markers are visible,
   - heatmap is visible with blue-to-red gradient,
   - clicking a marker shows disease label and confidence.

## Map behavior checks

1. Base layer is satellite.
2. Imagery appears grayscale/high-contrast.
3. Pan/zoom gestures are locked.
4. Overlays (markers + heatmap) remain bright enough to stand out.

## Regression checks

1. Reload app with no injected logs; map still loads.
2. Inject and refresh multiple times; map still renders and updates.
3. Verify no `Map rendering error` after repeated refreshes.
