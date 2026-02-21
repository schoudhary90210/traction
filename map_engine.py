#!/usr/bin/env python3
"""
================================================================================
 Agri-Scout — Map Engine  (Ghost Tractor Simulator + Google Maps Renderer)
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
    ):
        self.start_lat = start_lat
        self.start_lng = start_lng
        self.speed_mps = speed_mph * 0.44704
        self.speed_mph = speed_mph
        self._m_per_deg_lng = self._M_PER_DEG_LAT * math.cos(math.radians(start_lat))
        self._start_time = time.time()
        self._path_history = []
        self._total_rows = int(self.FIELD_HEIGHT_M / self.ROW_SPACING_M)

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

_MAP_STYLE = '[{"elementType":"geometry","stylers":[{"color":"#0e1117"}]},{"elementType":"labels.icon","stylers":[{"visibility":"off"}]},{"elementType":"labels.text.fill","stylers":[{"color":"#5a6072"}]},{"elementType":"labels.text.stroke","stylers":[{"color":"#0e1117"}]},{"featureType":"administrative","elementType":"geometry","stylers":[{"color":"#1a1d28"}]},{"featureType":"poi","elementType":"geometry","stylers":[{"color":"#12141c"}]},{"featureType":"road","elementType":"geometry.fill","stylers":[{"color":"#1a1f2e"}]},{"featureType":"road","elementType":"geometry.stroke","stylers":[{"color":"#0e1117"}]},{"featureType":"road.highway","elementType":"geometry.fill","stylers":[{"color":"#1e2538"}]},{"featureType":"transit","elementType":"geometry","stylers":[{"color":"#12141c"}]},{"featureType":"water","elementType":"geometry","stylers":[{"color":"#080a10"}]}]'


def generate_map_html(
    api_key,
    current_pos,
    heading,
    path_history,
    disease_logs,
    field_center,
    field_bounds,
    map_height=420,
):
    """
    Return self-contained HTML for a dark-mode Google Map with tractor marker,
    path trail, disease markers, and heatmap layer.
    """
    clat, clng = field_center
    tlat, tlng = current_pos
    nw_lat, nw_lng, se_lat, se_lng = field_bounds

    # Serialise path for JS
    path_parts = []
    for p in path_history[-300:]:
        path_parts.append("{lat:" + str(p[0]) + ",lng:" + str(p[1]) + "}")
    path_js = ",".join(path_parts)

    # Serialise disease logs for JS
    disease_parts = []
    for d in disease_logs:
        safe = d.label.replace("'", "\\'").replace('"', '\\"')
        disease_parts.append(
            "{lat:" + str(d.lat) + ",lng:" + str(d.lng)
            + ",label:'" + safe + "',conf:" + str(round(d.confidence, 2)) + "}"
        )
    disease_js = ",".join(disease_parts)

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
        '</style></head><body><div id="map"></div><script>'
        "function initMap(){"
        "var s=" + _MAP_STYLE + ";"
        "var m=new google.maps.Map(document.getElementById('map'),{"
        "center:{lat:" + str(clat) + ",lng:" + str(clng) + "},"
        "zoom:16,styles:s,disableDefaultUI:true,zoomControl:true,"
        "mapTypeId:'roadmap',backgroundColor:'#08090c'});"

        # Field boundary
        "new google.maps.Rectangle({bounds:{"
        "north:" + str(nw_lat) + ",south:" + str(se_lat) + ","
        "east:" + str(se_lng) + ",west:" + str(nw_lng) + "},"
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
        "title:'Agri-Scout',zIndex:999});"
        "var ti=new google.maps.InfoWindow({"
        "content:'<div class=\"tl\">\\ud83d\\ude9c AGRI-SCOUT</div>',"
        "disableAutoPan:true});"
        "ti.open(m,tm);"

        # Disease markers + heatmap
        "var ds=[" + disease_js + "];var hd=[];"
        "ds.forEach(function(d){"
        "var p=new google.maps.LatLng(d.lat,d.lng);"
        "hd.push({location:p,weight:d.conf});"
        "var c=new google.maps.Marker({position:p,map:m,"
        "icon:{path:google.maps.SymbolPath.CIRCLE,scale:10,"
        "fillColor:'#ff3d3d',fillOpacity:0.55,"
        "strokeColor:'#ff6b6b',strokeWeight:2,strokeOpacity:0.8},zIndex:500});"
        "new google.maps.Marker({position:p,map:m,"
        "icon:{path:google.maps.SymbolPath.CIRCLE,scale:4,"
        "fillColor:'#ff3d3d',fillOpacity:1,strokeWeight:0},zIndex:501});"
        "var pw=new google.maps.InfoWindow({"
        "content:'<div class=\"dp\"><b>\\u26a0 '+d.label+'</b><br>"
        "<i>Conf: '+(d.conf*100).toFixed(1)+'%</i><br>"
        "<i>'+d.lat.toFixed(5)+'\\u00b0N, '+Math.abs(d.lng).toFixed(5)+'\\u00b0W</i></div>'});"
        "c.addListener('click',function(){pw.open(m,c)});});"

        # Heatmap
        "if(hd.length>0){new google.maps.visualization.HeatmapLayer({"
        "data:hd,map:m,radius:35,opacity:0.5,"
        "gradient:['rgba(0,0,0,0)','rgba(255,61,61,0.2)','rgba(255,61,61,0.4)',"
        "'rgba(255,100,100,0.6)','rgba(255,140,60,0.8)','rgba(255,200,60,1)']});}"
        "}"

        "</script>"
        '<script async defer src="https://maps.googleapis.com/maps/api/js?key='
        + api_key + '&libraries=visualization&callback=initMap"></script>'
        "</body></html>"
    )