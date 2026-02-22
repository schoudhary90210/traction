#!/usr/bin/env python3
"""
================================================================================
 TRACTION — Map Engine  (Ghost Tractor Simulator + Google Maps Renderer)
================================================================================
 Zero external dependencies beyond Python stdlib.

 Provides:
   TractorSimulator  — Simulates a tractor driving a lawnmower grid at ~5 MPH.
   DiseaseLog        — Dataclass for geo-tagged disease events.
   generate_map_html — Returns HTML string for a dark-mode Google Map.
================================================================================
"""

import math
import time
import json
from dataclasses import dataclass
from typing import List, Tuple


# ==============================================================================
#  Data Types
# ==============================================================================

@dataclass
class DiseaseLog:
    """A single geo-tagged disease detection event."""
    lat: float
    lng: float
    label: str
    confidence: float
    timestamp: float


# ==============================================================================
#  Tractor Simulator
# ==============================================================================

class TractorSimulator:
    """
    Simulates a tractor driving a lawnmower (boustrophedon) grid pattern
    across a 140-acre rectangular field near Arena, Wisconsin.

    Call get_current_position() on each Streamlit rerun to advance the
    simulation based on wall-clock elapsed time.
    """

    DEFAULT_START_LAT = 43.1580
    DEFAULT_START_LNG = -89.9120

    FIELD_WIDTH_M  = 800.0
    FIELD_HEIGHT_M = 708.0
    ROW_SPACING_M  = 12.0
    SPEED_MPH      = 5.0

    _M_PER_DEG_LAT = 111_320.0

    def __init__(
        self,
        start_lat: float = DEFAULT_START_LAT,
        start_lng: float = DEFAULT_START_LNG,
        speed_mph: float = SPEED_MPH,
        field_width_m: float = FIELD_WIDTH_M,
        field_height_m: float = FIELD_HEIGHT_M,
        row_spacing_m: float = ROW_SPACING_M,
    ):
        self.start_lat = start_lat
        self.start_lng = start_lng
        self.speed_mps = speed_mph * 0.44704
        self.speed_mph = speed_mph
        self.FIELD_WIDTH_M = max(40.0, float(field_width_m))
        self.FIELD_HEIGHT_M = max(40.0, float(field_height_m))
        self.ROW_SPACING_M = max(2.0, min(float(row_spacing_m), self.FIELD_HEIGHT_M))
        self._m_per_deg_lng = self._M_PER_DEG_LAT * math.cos(math.radians(start_lat))
        self._start_time = time.time()
        self._path_history = []
        self._total_rows = max(1, int(self.FIELD_HEIGHT_M / self.ROW_SPACING_M))

    def get_current_position(self):
        """Returns (latitude, longitude, heading_degrees)."""
        elapsed = time.time() - self._start_time
        total_distance = elapsed * self.speed_mps

        row_length = self.FIELD_WIDTH_M
        transition = self.ROW_SPACING_M
        pass_len = row_length + transition

        full_passes = int(total_distance / pass_len)
        remainder = total_distance - (full_passes * pass_len)

        row_idx = full_passes % self._total_rows
        going_east = (row_idx % 2 == 0)

        if remainder <= row_length:
            frac = remainder / row_length
            x_offset = frac * row_length if going_east else (1.0 - frac) * row_length
            y_offset = row_idx * self.ROW_SPACING_M
            heading = 90.0 if going_east else 270.0
        else:
            trans_progress = remainder - row_length
            x_offset = row_length if going_east else 0.0
            y_offset = row_idx * self.ROW_SPACING_M + trans_progress
            heading = 180.0

        lat = self.start_lat - (y_offset / self._M_PER_DEG_LAT)
        lng = self.start_lng + (x_offset / self._m_per_deg_lng)

        pos = (round(lat, 7), round(lng, 7))
        if not self._path_history or self._path_history[-1] != pos:
            self._path_history.append(pos)
            if len(self._path_history) > 600:
                self._path_history = self._path_history[-400:]

        return lat, lng, heading

    @property
    def path_history(self):
        return self._path_history

    def get_speed_mph(self):
        return self.speed_mph

    def get_field_center(self):
        """Return (center_lat, center_lng) of the field."""
        clat = self.start_lat - (self.FIELD_HEIGHT_M / 2 / self._M_PER_DEG_LAT)
        clng = self.start_lng + (self.FIELD_WIDTH_M / 2 / self._m_per_deg_lng)
        return clat, clng

    def get_field_bounds(self):
        """Return (nw_lat, nw_lng, se_lat, se_lng) rectangle."""
        nw_lat = self.start_lat
        nw_lng = self.start_lng
        se_lat = nw_lat - (self.FIELD_HEIGHT_M / self._M_PER_DEG_LAT)
        se_lng = nw_lng + (self.FIELD_WIDTH_M / self._m_per_deg_lng)
        return nw_lat, nw_lng, se_lat, se_lng


# ==============================================================================
#  Google Maps HTML Generator
# ==============================================================================

_MAP_STYLE = '[{"stylers":[{"saturation":-100}]},{"featureType":"administrative","elementType":"labels.text.fill","stylers":[{"color":"#f0f0f0"}]},{"featureType":"road","elementType":"labels.text.fill","stylers":[{"color":"#d8d8d8"}]}]'


def generate_map_html(
    api_key,
    current_pos,
    heading,
    path_history,
    disease_logs,
    field_center,
    field_bounds,
    map_height=420,
    map_interactive=False,
    map_zoom=16,
    map_type="satellite",
    spot_scale=10,
):
    """
    Return self-contained HTML for a dark-mode Google Map with tractor marker,
    path trail, disease markers, and heatmap layer.
    """
    clat, clng = field_center
    tlat, tlng = current_pos
    nw_lat, nw_lng, se_lat, se_lng = field_bounds

    # Keep coordinates in web-mercator-safe ranges for stable rendering.
    clat = max(-85.0, min(85.0, float(clat)))
    tlat = max(-85.0, min(85.0, float(tlat)))
    nw_lat = max(-85.0, min(85.0, float(nw_lat)))
    se_lat = max(-85.0, min(85.0, float(se_lat)))
    clng = max(-179.9999, min(179.9999, float(clng)))
    tlng = max(-179.9999, min(179.9999, float(tlng)))
    nw_lng = max(-179.9999, min(179.9999, float(nw_lng)))
    se_lng = max(-179.9999, min(179.9999, float(se_lng)))

    # Ensure bounds ordering remains valid after clamping.
    north = max(nw_lat, se_lat)
    south = min(nw_lat, se_lat)
    east = max(nw_lng, se_lng)
    west = min(nw_lng, se_lng)

    safe_map_type = str(map_type).lower().strip()
    if safe_map_type not in {"roadmap", "hybrid", "satellite", "terrain"}:
        safe_map_type = "satellite"
    safe_spot_scale = max(3.0, min(24.0, float(spot_scale)))
    inner_spot_scale = max(1.5, safe_spot_scale * 0.42)
    safe_zoom = max(1, min(22, int(map_zoom)))
    disable_ui = "false" if map_interactive else "true"
    zoom_control = "true" if map_interactive else "false"
    gesture_handling = "'greedy'" if map_interactive else "'none'"

    # Serialise path for JS
    path_parts = []
    for p in path_history[-300:]:
        path_parts.append("{lat:" + str(p[0]) + ",lng:" + str(p[1]) + "}")
    path_js = ",".join(path_parts)

    # Serialise disease logs for JS
    disease_parts = []
    for d in disease_logs:
        safe = d.label.replace("'", "\\'").replace('"', '\\"')
        conf = max(0.0, min(1.0, float(d.confidence)))
        disease_parts.append(
            "{lat:" + str(d.lat) + ",lng:" + str(d.lng)
            + ",label:'" + safe + "',conf:" + str(conf) + "}"
        )
    disease_js = ",".join(disease_parts)

    if not str(api_key or "").strip():
        tile_url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        tile_attr = "&copy; OpenStreetMap contributors"
        if safe_map_type in {"satellite", "hybrid"}:
            tile_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            tile_attr = "Tiles &copy; Esri"
        elif safe_map_type == "terrain":
            tile_url = "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
            tile_attr = "Map data: &copy; OpenStreetMap contributors, SRTM"

        path_json = json.dumps(
            [
                [max(-85.0, min(85.0, float(p[0]))), max(-179.9999, min(179.9999, float(p[1])))]
                for p in path_history[-300:]
            ]
        )
        disease_json = json.dumps(
            [
                {
                    "lat": max(-85.0, min(85.0, float(d.lat))),
                    "lng": max(-179.9999, min(179.9999, float(d.lng))),
                    "label": str(d.label),
                    "conf": max(0.0, min(1.0, float(d.confidence))),
                }
                for d in disease_logs
            ]
        )
        return (
            '<!DOCTYPE html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1"/>'
            '<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>'
            "<style>"
            "*{margin:0;padding:0;box-sizing:border-box}"
            "html,body{background:#08090c}"
            "#map{width:100%;height:" + str(map_height) + "px;border-radius:10px}"
            ".leaflet-container{background:#08090c;font-family:monospace}"
            ".tractor-icon{width:22px;height:22px;display:flex;align-items:center;justify-content:center}"
            ".tractor-arrow{width:0;height:0;border-left:10px solid transparent;border-right:10px solid transparent;"
            "border-bottom:18px solid #00e676;filter:drop-shadow(0 0 3px rgba(0,0,0,0.7));"
            "transform-origin:50% 66%;}"
            ".leaflet-popup-content-wrapper{background:#10121a;color:#eaedf3;border:1px solid #25304b;border-radius:8px}"
            ".leaflet-popup-tip{background:#10121a}"
            "</style></head><body>"
            '<div id="map"></div>'
            '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>'
            "<script>"
            "const pathData=" + path_json + ";"
            "const diseaseData=" + disease_json + ";"
            "const map=L.map('map',{zoomControl:" + ("true" if map_interactive else "false")
            + ",dragging:true,scrollWheelZoom:true,doubleClickZoom:true,boxZoom:true,keyboard:true,tap:true});"
            "L.tileLayer('" + tile_url + "',{attribution:'" + tile_attr + "',maxZoom:20}).addTo(map);"
            "const bounds=L.latLngBounds([[" + str(south) + "," + str(west) + "],[" + str(north) + "," + str(east) + "]]);"
            "if(bounds.isValid()){map.fitBounds(bounds.pad(0.06));}else{map.setView([" + str(clat) + "," + str(clng) + "]," + str(safe_zoom) + ");}"
            "if(!" + ("true" if map_interactive else "false") + "){"
            "map.dragging.disable();map.scrollWheelZoom.disable();map.doubleClickZoom.disable();"
            "map.boxZoom.disable();map.keyboard.disable();map.touchZoom.disable();}"
            "L.rectangle([[" + str(south) + "," + str(west) + "],[" + str(north) + "," + str(east)
            + "]],{color:'#1a4028',weight:1.5,opacity:0.7,fillColor:'#0a1f12',fillOpacity:0.18}).addTo(map);"
            "if(pathData.length>1){L.polyline(pathData,{color:'#00e676',weight:2.5,opacity:0.5}).addTo(map);}"
            "const tractorIcon=L.divIcon({className:'tractor-icon',iconSize:[22,22],iconAnchor:[11,11],"
            "html:'<div class=\"tractor-arrow\" style=\"transform:rotate(" + str(float(heading)) + "deg)\"></div>'});"
            "const tractorMarker=L.marker([" + str(tlat) + "," + str(tlng) + "],{icon:tractorIcon,zIndexOffset:900}).addTo(map);"
            "tractorMarker.bindPopup('🚜 TRACTION').openPopup();"
            "diseaseData.forEach((d)=>{"
            "const conf=Math.max(0,Math.min(1,Number(d.conf||0)));"
            "const pos=[d.lat,d.lng];"
            "const glowRadius=Math.max(8,Math.round(conf*" + str(max(20.0, safe_spot_scale * 8.0)) + "));"
            "L.circle(pos,{radius:glowRadius,color:'#ff6b6b',weight:1,opacity:0.3,fillColor:'#ff3d3d',fillOpacity:0.16}).addTo(map);"
            "const marker=L.circleMarker(pos,{radius:" + str(safe_spot_scale) + ",color:'#ffffff',weight:1.2,fillColor:'#ff3d3d',fillOpacity:0.88}).addTo(map);"
            "marker.bindPopup('<b>⚠ '+String(d.label||'Detection')+'</b><br/>Conf: '+(conf*100).toFixed(1)+'%');"
            "});"
            "</script></body></html>"
        )

    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><style>'
        '*{margin:0;padding:0;box-sizing:border-box}'
        'html,body{background:#08090c}'
        '#map{width:100%;height:' + str(map_height) + 'px;border-radius:10px}'
        ".dp{font-family:monospace;background:#10121a;border:1px solid #2a1a1a;"
        "border-radius:8px;padding:8px 12px;color:#eaedf3;font-size:11px}"
        ".dp b{color:#ff6b6b;font-size:12px}"
        ".dp i{color:#5a6072;font-style:normal}"
        ".tl{background:rgba(0,230,118,0.15);border:1px solid rgba(0,230,118,0.35);"
        "border-radius:6px;padding:2px 8px;font-family:monospace;font-size:10px;"
        "color:#00e676;font-weight:600}"
        '</style><script>'
        "function initMap(){"
        "var s=" + _MAP_STYLE + ";"
        "var m=new google.maps.Map(document.getElementById('map'),{"
        "center:{lat:" + str(clat) + ",lng:" + str(clng) + "},"
        "zoom:" + str(safe_zoom) + ",styles:s,disableDefaultUI:" + disable_ui + ",zoomControl:" + zoom_control + ","
        "gestureHandling:" + gesture_handling + ",mapTypeId:'" + safe_map_type + "',backgroundColor:'#08090c'});"
        "var fieldBounds={north:" + str(north) + ",south:" + str(south) + ",east:" + str(east) + ",west:" + str(west) + "};"
        "m.fitBounds(fieldBounds);"
        "google.maps.event.addListenerOnce(m,'bounds_changed',function(){if(m.getZoom()>18){m.setZoom(18);}});"

        # Field boundary
        "new google.maps.Rectangle({bounds:{"
        "north:" + str(north) + ",south:" + str(south) + ","
        "east:" + str(east) + ",west:" + str(west) + "},"
        "map:m,strokeColor:'#1a4028',strokeOpacity:0.6,strokeWeight:1.5,"
        "fillColor:'#0a1f12',fillOpacity:0.15});"

        # Path trail
        "var pc=[" + path_js + "];"
        "if(pc.length>1){new google.maps.Polyline({"
        "path:pc,map:m,strokeColor:'#00e676',strokeOpacity:0.35,strokeWeight:2.5});}"

        # Tractor marker
        "var tp={lat:" + str(tlat) + ",lng:" + str(tlng) + "};"
        "var tm=new google.maps.Marker({position:tp,map:m,"
        "icon:{path:google.maps.SymbolPath.FORWARD_CLOSED_ARROW,"
        "scale:7,fillColor:'#00e676',fillOpacity:1,"
        "strokeColor:'#08090c',strokeWeight:2,"
        "rotation:" + str(heading) + ","
        "anchor:new google.maps.Point(0,2.5)},"
        "title:'TRACTION',zIndex:999});"
        "var ti=new google.maps.InfoWindow({"
        "content:'<div class=\"tl\">\\ud83d\\ude9c TRACTION</div>',"
        "disableAutoPan:true});"
        "ti.open(m,tm);"

        # Disease markers + heatmap
        "var ds=[" + disease_js + "];var hd=[];"
        "ds.forEach(function(d){"
        "var p=new google.maps.LatLng(d.lat,d.lng);"
        "/** @type {google.maps.visualization.WeightedLocation} */"
        "var wl={location:p,weight:d.conf};hd.push(wl);"
        "var c=new google.maps.Marker({position:p,map:m,"
        "icon:{path:google.maps.SymbolPath.CIRCLE,scale:" + str(safe_spot_scale) + ","
        "fillColor:'#ff3d3d',fillOpacity:0.55,"
        "strokeColor:'#ffffff',strokeWeight:2,strokeOpacity:0.95},zIndex:500});"
        "new google.maps.Marker({position:p,map:m,"
        "icon:{path:google.maps.SymbolPath.CIRCLE,scale:" + str(inner_spot_scale) + ","
        "fillColor:'#ff3d3d',fillOpacity:1,strokeWeight:0},zIndex:501});"
        "var pw=new google.maps.InfoWindow({"
        "content:'<div class=\"dp\"><b>\\u26a0 '+d.label+'</b><br>"
        "<i>Conf: '+(d.conf*100).toFixed(1)+'%</i><br>"
        "<i>'+d.lat.toFixed(5)+'\\u00b0N, '+Math.abs(d.lng).toFixed(5)+'\\u00b0W</i></div>'});"
        "c.addListener('click',function(){pw.open(m,c)});});"

        # Heatmap
        "var heatmap=new google.maps.visualization.HeatmapLayer({"
        "data:hd,map:m,radius:" + str(int(max(16.0, min(64.0, safe_spot_scale * 3.0)))) + ",opacity:0.75,"
        "gradient:['rgba(0,0,0,0)','rgba(50,114,255,0.30)','rgba(50,114,255,0.55)',"
        "'rgba(50,114,255,0.80)','rgba(255,82,82,0.95)','rgba(255,61,61,1.00)']});"
        "}"

        "</script>"
        '<script defer src="https://maps.googleapis.com/maps/api/js?key='
        + api_key + '&libraries=visualization&callback=initMap"></script>'
        "</head><body><div id=\"map\"></div></body></html>"
    )
