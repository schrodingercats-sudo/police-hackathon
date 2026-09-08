/**
 * MapView - Leaflet 1.9.4 Interactive Police GIS Map Visualizer
 * Plots CCTV camera nodes, target detection waypoints, directional route trajectories,
 * speed feasibility overlays, and pulsing radar tracking beacons.
 */

const MapView = (function () {
  let map = null;
  let tileLayers = {};
  let cameraLayer = null;
  let waypointLayer = null;
  let routeLayer = null;
  let theftOriginLayer = null;

  let currentWaypoints = [];
  let waypointMarkersMap = new Map();
  let currentBounds = null;
  let waypointCallback = null;

  /**
   * Initialize Leaflet Map instance with dark command center basemap
   */
  function initMap(containerId = 'map') {
    if (map) return map;

    // Center on Delhi NCR surveillance corridor
    const defaultCenter = [28.6100, 77.2700];
    const defaultZoom = 12;

    map = L.map(containerId, {
      center: defaultCenter,
      zoom: defaultZoom,
      zoomControl: false,
      attributionControl: false
    });

    // Add zoom control on top right
    L.control.zoom({ position: 'topright' }).addTo(map);

    // Basemaps
    tileLayers.dark = L.tileLayer(
      'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
      {
        maxZoom: 19,
        subdomains: 'abcd',
        attribution: '&copy; CartoDB &copy; OpenStreetMap'
      }
    );

    tileLayers.satellite = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      {
        maxZoom: 18,
        attribution: '&copy; Esri &copy; DigitalGlobe',
        errorTileUrl: 'https://tile.openstreetmap.org/1/0/0.png'
      }
    );

    tileLayers.streets = L.tileLayer(
      'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      {
        maxZoom: 19,
        attribution: '&copy; OpenStreetMap contributors'
      }
    );

    // Default dark basemap
    tileLayers.dark.addTo(map);

    // Initialize overlay layer groups
    cameraLayer = L.layerGroup().addTo(map);
    waypointLayer = L.layerGroup().addTo(map);
    routeLayer = L.layerGroup().addTo(map);
    theftOriginLayer = L.layerGroup().addTo(map);

    return map;
  }

  /**
   * Plot registered CCTV Surveillance Camera nodes on map
   */
  function plotCameras(cameraList) {
    if (!cameraLayer) return;
    cameraLayer.clearLayers();

    if (!cameraList || !Array.isArray(cameraList)) return;

    cameraList.forEach((cam) => {
      if (!cam.latitude || !cam.longitude) return;

      const iconHtml = `<div class="custom-camera-marker" title="${cam.name || cam.id}">
          <i class="fa-solid fa-video"></i>
        </div>`;

      const cameraIcon = L.divIcon({
        className: 'leaflet-camera-div-icon',
        html: iconHtml,
        iconSize: [32, 32],
        iconAnchor: [16, 16],
        popupAnchor: [0, -18]
      });

      const marker = L.marker([cam.latitude, cam.longitude], { icon: cameraIcon });

      const statusColor = cam.status === 'active' ? '#10b981' : '#f59e0b';
      const statusText = cam.status === 'active' ? 'ONLINE' : 'OFFLINE';
      const camType = (cam.camera_type || cam.type || 'urban').toUpperCase();
      const bearing = cam.direction_bearing || cam.bearing || 0;
      const detections = cam.detections_today || 0;

      const popupHtml = `<div style="font-family: 'Inter', sans-serif; min-width: 240px;">
          <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #38bdf8;">
              <i class="fa-solid fa-camera"></i> ${cam.id}
            </span>
            <span style="font-size: 10px; font-weight: 700; text-transform: uppercase; color: ${statusColor};">
              &#9679; ${statusText}
            </span>
          </div>
          <div style="font-size: 13px; font-weight: 700; color: #fff; margin-bottom: 4px;">
            ${cam.name || cam.id}
          </div>
          <div style="font-size: 11px; color: #94a3b8; margin-bottom: 8px;">
            <i class="fa-solid fa-location-dot"></i> ${cam.road_name || 'Delhi NCR Corridor'}
          </div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 10px; background: rgba(15,23,42,0.6); padding: 6px; border-radius: 4px; border: 1px solid #1e293b; color: #e2e8f0;">
            <div><strong style="color: #64748b;">TYPE:</strong> ${camType}</div>
            <div><strong style="color: #64748b;">BEARING:</strong> ${bearing}&deg;</div>
            <div><strong style="color: #64748b;">LAT:</strong> ${cam.latitude.toFixed(4)}</div>
            <div><strong style="color: #64748b;">LON:</strong> ${cam.longitude.toFixed(4)}</div>
          </div>
        </div>`;

      marker.bindPopup(popupHtml);
      cameraLayer.addLayer(marker);
    });
  }

  /**
   * Plot reported theft origin marker
   */
  function plotTheftOrigin(theftData) {
    if (!theftOriginLayer) return;
    theftOriginLayer.clearLayers();

    if (!theftData || !theftData.latitude || !theftData.longitude) return;

    const iconHtml = `<div style="
        width: 32px; height: 32px; border-radius: 50%;
        background: #dc2626; border: 2px solid #fff;
        display: flex; align-items: center; justify-content: center;
        color: #fff; font-size: 14px; box-shadow: 0 0 16px rgba(220, 38, 38, 0.8);
      " title="Theft Origin Point">
        <i class="fa-solid fa-triangle-exclamation"></i>
      </div>`;

    const theftIcon = L.divIcon({
      className: 'leaflet-theft-div-icon',
      html: iconHtml,
      iconSize: [32, 32],
      iconAnchor: [16, 16],
      popupAnchor: [0, -18]
    });

    const marker = L.marker([theftData.latitude, theftData.longitude], { icon: theftIcon });

    const locName = theftData.location_name || 'Reported Location';
    const dtStr = theftData.datetime || '';

    const popupHtml = `<div style="font-family: 'Inter', sans-serif; min-width: 220px;">
        <div style="font-size: 11px; font-weight: 700; color: #ef4444; text-transform: uppercase; margin-bottom: 4px;">
          <i class="fa-solid fa-triangle-exclamation"></i> Reported Theft Origin
        </div>
        <div style="font-size: 13px; font-weight: 700; color: #fff; margin-bottom: 4px;">
          ${locName}
        </div>
        <div style="font-size: 11px; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
          Time: ${dtStr}
        </div>
      </div>`;

    marker.bindPopup(popupHtml);
    theftOriginLayer.addLayer(marker);
  }

  /**
   * Helper: Calculate bearing heading between two lat/lons in degrees
   */
  function calculateBearing(lat1, lon1, lat2, lon2) {
    const dLon = (lon2 - lon1) * (Math.PI / 180);
    const y = Math.sin(dLon) * Math.cos(lat2 * (Math.PI / 180));
    const x =
      Math.cos(lat1 * (Math.PI / 180)) * Math.sin(lat2 * (Math.PI / 180)) -
      Math.sin(lat1 * (Math.PI / 180)) * Math.cos(lat2 * (Math.PI / 180)) * Math.cos(dLon);
    let brng = Math.atan2(y, x) * (180 / Math.PI);
    return (brng + 360) % 360;
  }

  /**
   * Plot chronological route waypoints and connecting directional polyline segments
   */
  function plotRoute(routeData, onWaypointClick) {
    waypointCallback = onWaypointClick;
    if (!waypointLayer || !routeLayer) return;

    waypointLayer.clearLayers();
    routeLayer.clearLayers();
    waypointMarkersMap.clear();

    if (!routeData) return;

    // 1. Plot Theft Origin
    if (routeData.theft_origin) {
      plotTheftOrigin(routeData.theft_origin);
    }

    const waypoints = routeData.waypoints || [];
    const segments = routeData.segments || [];
    currentWaypoints = waypoints;

    const latLngList = [];

    if (routeData.theft_origin && routeData.theft_origin.latitude) {
      latLngList.push([routeData.theft_origin.latitude, routeData.theft_origin.longitude]);
    }

    // 2. Plot Waypoint Nodes
    waypoints.forEach((wp, idx) => {
      if (!wp.latitude || !wp.longitude) return;

      const latLng = [wp.latitude, wp.longitude];
      latLngList.push(latLng);

      const isLast = idx === waypoints.length - 1;
      const statusClass =
        wp.verification_status === 'verified'
          ? 'waypoint-verified'
          : wp.verification_status === 'rejected'
          ? 'waypoint-rejected'
          : 'waypoint-pending';

      const radarHtml = isLast ? `<div class="radar-beacon"></div>` : '';

      const iconHtml = `<div style="position: relative;">
          ${radarHtml}
          <div class="custom-waypoint-marker ${statusClass}">
            #${wp.sequence || idx + 1}
          </div>
        </div>`;

      const waypointIcon = L.divIcon({
        className: 'leaflet-waypoint-div-icon',
        html: iconHtml,
        iconSize: [28, 28],
        iconAnchor: [14, 14],
        popupAnchor: [0, -16]
      });

      const marker = L.marker(latLng, { icon: waypointIcon });

      const cropImgSrc = wp.vehicle_crop_url || '';
      const plateImgSrc = wp.plate_crop_url || '';
      const formattedTime = wp.timestamp
        ? new Date(wp.timestamp).toLocaleString('en-IN', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            day: '2-digit',
            month: 'short'
          })
        : 'N/A';

      const speedVal = wp.speed_from_prev_kmh;
      const distVal = wp.distance_from_prev_km;
      const speedInfo =
        speedVal !== null && speedVal !== undefined
          ? `<span style="color: #38bdf8;">Speed: ${speedVal} km/h (${distVal || '?'} km)</span>`
          : `<span style="color: #94a3b8;">First Sighting Lead</span>`;

      const vStatus = wp.verification_status || 'pending';
      const statusBg = vStatus === 'verified' ? 'background:#065f46;color:#34d399;' : vStatus === 'rejected' ? 'background:#7f1d1d;color:#fca5a5;' : 'background:#78350f;color:#fbbf24;';
      const cScore = wp.composite_score ? (wp.composite_score * 100).toFixed(0) : '90';
      const pConf = wp.plate_confidence ? (wp.plate_confidence * 100).toFixed(0) : '95';
      const camName = wp.camera_name || wp.camera_id || '';
      const roadName = wp.road_name || '';
      const plateText = wp.plate_text || '';

      const placeholderSvg = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='100' height='65' fill='%231e293b'><rect width='100' height='65'/><text x='50%25' y='50%25' fill='%2364748b' text-anchor='middle' font-size='10'>No Image</text></svg>";

      const popupHtml = `<div style="font-family: 'Inter', sans-serif; min-width: 260px;">
          <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 800; color: #fff;">
              SIGHTING #${wp.sequence || idx + 1}
            </span>
            <span style="font-size: 10px; font-weight: 700; text-transform: uppercase; padding: 1px 6px; border-radius: 3px; ${statusBg}">
              ${vStatus}
            </span>
          </div>

          <div style="font-size: 12px; font-weight: 700; color: #fff; margin-bottom: 2px;">
            ${camName}
          </div>
          <div style="font-size: 10px; color: #94a3b8; margin-bottom: 8px;">
            <i class="fa-solid fa-road"></i> ${roadName}
          </div>

          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 8px;">
            <div style="height: 65px; background: #000; border-radius: 4px; overflow: hidden; border: 1px solid #334155;">
              <img src="${cropImgSrc}" style="width: 100%; height: 100%; object-fit: cover;" onerror="this.src='${placeholderSvg}';" />
            </div>
            <div style="height: 65px; background: #000; border-radius: 4px; overflow: hidden; border: 1px solid #334155; position: relative;">
              <img src="${plateImgSrc}" style="width: 100%; height: 100%; object-fit: cover;" onerror="this.src='${placeholderSvg}';" />
              <div style="position: absolute; bottom: 2px; left: 2px; right: 2px; background: rgba(0,0,0,0.8); color: #fbbf24; font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: bold; text-align: center;">
                ${plateText}
              </div>
            </div>
          </div>

          <div style="font-size: 10px; font-family: 'JetBrains Mono', monospace; display: flex; flex-direction: column; gap: 2px; margin-bottom: 6px; background: rgba(15,23,42,0.7); padding: 6px; border-radius: 4px; color: #e2e8f0;">
            <div><i class="fa-regular fa-clock"></i> ${formattedTime}</div>
            <div><i class="fa-solid fa-gauge-high"></i> ${speedInfo}</div>
            <div><i class="fa-solid fa-bullseye"></i> Match Score: <strong style="color: #10b981;">${cScore}%</strong> (OCR Conf: ${pConf}%)</div>
          </div>
        </div>`;

      marker.bindPopup(popupHtml);

      marker.on('click', () => {
        if (typeof waypointCallback === 'function') {
          waypointCallback(wp.sighting_id);
        }
      });

      waypointLayer.addLayer(marker);
      waypointMarkersMap.set(wp.sighting_id, marker);
    });

    // 3. Draw Connecting Route Segments with Speed Feasibility Styling & Directional Arrowheads
    if (segments.length > 0) {
      segments.forEach((seg) => {
        if (!seg.from_coords || !seg.to_coords) return;

        const isFeasible = seg.feasible !== false;
        const color = isFeasible ? '#38bdf8' : '#ef4444';
        const dashArray = isFeasible ? null : '6, 8';

        const polyline = L.polyline([seg.from_coords, seg.to_coords], {
          color: color,
          weight: 4,
          opacity: 0.85,
          dashArray: dashArray,
          lineJoin: 'round'
        });

        const elapsedMin = seg.elapsed_time_sec ? (seg.elapsed_time_sec / 60).toFixed(1) : '?';
        const feasibleText = isFeasible ? 'FEASIBLE' : 'INFEASIBLE';
        const feasibleColor = isFeasible ? '#10b981' : '#ef4444';

        const segPopup = `<div style="font-family: 'Inter', sans-serif; font-size: 11px; color: #e2e8f0;">
            <strong style="color: #fff;">Transit Segment:</strong> ${seg.from_camera_id || '?'} &rarr; ${seg.to_camera_id || '?'}<br/>
            <strong>Distance:</strong> ${seg.distance_km || '?'} km<br/>
            <strong>Elapsed Time:</strong> ${elapsedMin} min (${seg.elapsed_time_sec || '?'}s)<br/>
            <strong>Speed:</strong> <span style="font-weight: bold; color: ${color};">${seg.speed_kmh || '?'} km/h</span><br/>
            <strong>Feasibility:</strong> <span style="font-weight: bold; color: ${feasibleColor};">${feasibleText}</span>
          </div>`;
        polyline.bindPopup(segPopup);
        routeLayer.addLayer(polyline);

        // Add Directional Arrowhead marker at midpoint
        const midLat = (seg.from_coords[0] + seg.to_coords[0]) / 2;
        const midLon = (seg.from_coords[1] + seg.to_coords[1]) / 2;
        const bearing = calculateBearing(
          seg.from_coords[0],
          seg.from_coords[1],
          seg.to_coords[0],
          seg.to_coords[1]
        );

        const arrowHtml = `<div style="
            transform: rotate(${bearing}deg);
            color: ${color};
            font-size: 14px;
            text-shadow: 0 0 6px rgba(0,0,0,0.9);
            display: flex; align-items: center; justify-content: center;
          ">
            <i class="fa-solid fa-chevron-up"></i>
          </div>`;

        const arrowIcon = L.divIcon({
          className: 'leaflet-arrow-div-icon',
          html: arrowHtml,
          iconSize: [16, 16],
          iconAnchor: [8, 8]
        });

        const arrowMarker = L.marker([midLat, midLon], { icon: arrowIcon });
        arrowMarker.bindPopup(segPopup);
        routeLayer.addLayer(arrowMarker);
      });
    }

    // 4. Fit bounds to contain all waypoints
    if (latLngList.length > 0) {
      currentBounds = L.latLngBounds(latLngList);
      fitRouteBounds();
    }
  }

  /**
   * Fit map viewport to encompass active route
   */
  function fitRouteBounds() {
    if (map && currentBounds && currentBounds.isValid()) {
      map.fitBounds(currentBounds, {
        padding: [60, 60],
        maxZoom: 15,
        animate: true
      });
    }
  }

  /**
   * Highlight and pan to specific sighting marker when card is clicked
   */
  function highlightWaypoint(sightingId) {
    if (!map || !waypointMarkersMap.has(sightingId)) return;

    const marker = waypointMarkersMap.get(sightingId);
    if (marker) {
      map.panTo(marker.getLatLng(), { animate: true, duration: 0.5 });
      marker.openPopup();
    }
  }

  /**
   * Toggle visibility of specific layer group
   */
  function toggleLayer(layerName, isVisible) {
    if (!map) return;

    let targetLayer = null;
    if (layerName === 'cameras') targetLayer = cameraLayer;
    if (layerName === 'waypoints') targetLayer = waypointLayer;
    if (layerName === 'route') targetLayer = routeLayer;

    if (targetLayer) {
      if (isVisible) {
        if (!map.hasLayer(targetLayer)) map.addLayer(targetLayer);
      } else {
        if (map.hasLayer(targetLayer)) map.removeLayer(targetLayer);
      }
    }
  }

  /**
   * Switch base tile layer (dark, satellite, streets)
   */
  function switchBaseLayer(layerType) {
    if (!map || !tileLayers[layerType]) return;

    Object.values(tileLayers).forEach((layer) => {
      if (map.hasLayer(layer)) map.removeLayer(layer);
    });

    tileLayers[layerType].addTo(map);
  }

  /**
   * Pan and zoom map to coordinates (lat, lon)
   */
  function panToLocation(lat, lon, zoom = 15) {
    if (!map || lat == null || lon == null) return;
    map.setView([lat, lon], zoom, { animate: true, duration: 0.8 });
  }

  /**
   * Recalculate map container size after layout shifts
   */
  function invalidateSize() {
    if (map) {
      map.invalidateSize();
    }
  }

  return {
    initMap,
    plotCameras,
    plotTheftOrigin,
    plotRoute,
    highlightWaypoint,
    panToLocation,
    invalidateSize,
    fitRouteBounds,
    toggleLayer,
    switchBaseLayer
  };
})();

// Export globally
window.MapView = MapView;