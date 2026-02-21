import unittest

from map_engine import DiseaseLog, generate_map_html


def _build_html(disease_logs, **kwargs):
    return generate_map_html(
        api_key="test-key",
        current_pos=(43.1580, -89.9120),
        heading=90.0,
        path_history=[
            (43.1580, -89.9120),
            (43.1579, -89.9118),
        ],
        disease_logs=disease_logs,
        field_center=(43.1575, -89.9115),
        field_bounds=(43.1582, -89.9122, 43.1568, -89.9108),
        map_height=420,
        **kwargs,
    )


class HeatmapMapEngineTests(unittest.TestCase):
    def test_library_and_callback_wiring(self):
        html = _build_html([])
        self.assertIn("libraries=visualization", html)
        self.assertIn("callback=initMap", html)

    def test_heatmap_layer_initialization(self):
        html = _build_html([
            DiseaseLog(
                lat=43.1578,
                lng=-89.9117,
                label="Common Rust",
                confidence=0.92,
                timestamp=1_700_000_000.0,
            )
        ])
        self.assertIn("google.maps.visualization.HeatmapLayer", html)
        self.assertIn("radius:30", html)
        self.assertIn("WeightedLocation", html)
        self.assertIn("weight:d.conf", html)
        self.assertIn("rgba(50,114,255,0.30)", html)
        self.assertIn("rgba(255,61,61,1.00)", html)

    def test_map_mode_and_lock_options(self):
        html = _build_html([])
        self.assertIn("mapTypeId:'satellite'", html)
        self.assertIn("gestureHandling:'none'", html)
        self.assertIn("zoomControl:false", html)
        self.assertIn("disableDefaultUI:true", html)
        self.assertIn("saturation\":-100", html)

    def test_interactive_mode_enables_pan_and_zoom_controls(self):
        html = _build_html([], map_interactive=True, map_zoom=21)
        self.assertIn("zoom:21", html)
        self.assertIn("gestureHandling:'greedy'", html)
        self.assertIn("zoomControl:true", html)
        self.assertIn("disableDefaultUI:false", html)

    def test_disease_log_serialization_clamps_confidence_and_escapes_label(self):
        html = _build_html([
            DiseaseLog(
                lat=43.1577,
                lng=-89.9116,
                label="Low",
                confidence=-0.3,
                timestamp=1_700_000_001.0,
            ),
            DiseaseLog(
                lat=43.1576,
                lng=-89.9115,
                label="Mid",
                confidence=0.6,
                timestamp=1_700_000_002.0,
            ),
            DiseaseLog(
                lat=43.1575,
                lng=-89.9114,
                label="Rust 'Severe' \"Type\"",
                confidence=1.4,
                timestamp=1_700_000_003.0,
            ),
        ])

        self.assertIn("conf:0.0", html)
        self.assertIn("conf:0.6", html)
        self.assertIn("conf:1.0", html)
        self.assertNotIn("conf:-0.3", html)
        self.assertNotIn("conf:1.4", html)
        self.assertIn("label:'Rust \\'Severe\\' \\\"Type\\\"'", html)

    def test_empty_logs_still_generates_valid_map_html(self):
        html = _build_html([])
        self.assertIn("function initMap(){", html)
        self.assertIn("var ds=[];var hd=[];", html)
        self.assertIn("google.maps.visualization.HeatmapLayer", html)


if __name__ == "__main__":
    unittest.main()
