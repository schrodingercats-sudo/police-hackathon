/**
 * Police Command Dashboard - Main Application Controller
 * Manages operational state, telemetry polling, case routing, GIS map interactions,
 * multi-modal search dispatches, and FIR incident intake.
 */

const App = (function () {
  let activeCaseId = null;
  let activeCaseData = null;
  let allCases = [];
  let allCameras = [];
  let officerBadgeId = 'DL-4821';
  let statsPollInterval = null;

  /**
   * Application Initialization
   */
  async function init() {
    console.log('Initializing Police Command Center Dashboard...');

    // 1. Initialize Leaflet Map
    if (window.MapView && typeof window.MapView.initMap === 'function') {
      window.MapView.initMap('map');
    }

    // 2. Start Live Clocks
    startLiveClocks();

    // 3. Bind UI Events & Modals
    bindEventHandlers();

    // 4. Initial Telemetry & Cameras Fetch
    await fetchDashboardStats();
    await fetchCameras();

    // 5. Fetch Cases & Load Initial Target Case
    await fetchCases();

    // 6. Start Periodic Telemetry Polling (every 10s)
    statsPollInterval = setInterval(fetchDashboardStats, 10000);

    showToast('Command Dashboard Initialized • All 6 Surveillance Nodes Online', 'info');
  }

  /**
   * Live Digital Clocks (IST + UTC)
   */
  function startLiveClocks() {
    function updateClock() {
      const now = new Date();
      const istStr = now.toLocaleTimeString('en-IN', {
        timeZone: 'Asia/Kolkata',
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      });
      const clockElem = document.getElementById('live-clock-text');
      if (clockElem) {
        clockElem.innerText = `${istStr} IST`;
      }
    }
    updateClock();
    setInterval(updateClock, 1000);
  }

  /**
   * Fetch Dashboard Telemetry KPIs (GET /api/dashboard/stats)
   */
  async function fetchDashboardStats() {
    try {
      const res = await fetch('/api/dashboard/stats');
      if (!res.ok) return;
      const data = await res.json();

      updateKpiElement('kpi-active-cases', data.active_cases || 0);
      updateKpiElement(
        'kpi-cameras-online',
        `${data.active_cameras || 0}/${data.total_cameras || 0}`
      );
      updateKpiElement('kpi-detections-today', data.detections_today || 0);
      updateKpiElement('kpi-matches-today', data.matches_today || 0);
      updateKpiElement('kpi-recovery-rate', `${data.recovery_rate_pct || 0}%`);
    } catch (err) {
      console.warn('Telemetry polling error:', err);
    }
  }

  function updateKpiElement(id, value) {
    const el = document.getElementById(id);
    if (el) el.innerText = value;
  }

  /**
   * Fetch Camera Network Registry (GET /api/cameras)
   */
  async function fetchCameras() {
    try {
      const res = await fetch('/api/cameras');
      if (!res.ok) return;
      allCameras = await res.json();

      // Plot cameras on Map
      if (window.MapView && typeof window.MapView.plotCameras === 'function') {
        window.MapView.plotCameras(allCameras);
      }

      // Populate camera selectors in search modal
      populateCameraDropdowns(allCameras);
    } catch (err) {
      console.error('Failed to fetch cameras:', err);
    }
  }

  function populateCameraDropdowns(cameras) {
    const select = document.getElementById('search-camera-filter');
    if (!select) return;

    select.innerHTML = '<option value="">All Surveillance Nodes (Delhi NCR)</option>';
    cameras.forEach((cam) => {
      const opt = document.createElement('option');
      opt.value = cam.id;
      opt.innerText = `${cam.id} - ${cam.name}`;
      select.appendChild(opt);
    });
  }

  /**
   * Fetch All FIR Stolen Vehicle Cases (GET /api/cases)
   */
  async function fetchCases() {
    try {
      const res = await fetch('/api/cases');
      if (!res.ok) return;
      allCases = await res.json();

      const dropdown = document.getElementById('case-select-dropdown');
      if (dropdown) {
        dropdown.innerHTML = '';
        allCases.forEach((c) => {
          const opt = document.createElement('option');
          opt.value = c.id;
          opt.innerText = `${c.fir_number} [${c.reported_plate}] - ${c.vehicle_color || ''} ${c.make || ''} ${c.model || ''}`;
          dropdown.appendChild(opt);
        });
      }

      // If activeCaseId is not set, default to FIR-2026-DEL-0941 or first case
      if (!activeCaseId && allCases.length > 0) {
        const targetCase =
          allCases.find((c) => c.fir_number === 'FIR-2026-DEL-0941') || allCases[0];
        await loadCase(targetCase.id);
      }
    } catch (err) {
      console.error('Failed to fetch cases:', err);
    }
  }

  /**
   * Load Active Case & Reconstruct Trajectory Route
   */
  async function loadCase(caseId, isRefresh = false) {
    activeCaseId = caseId;

    // Update Dropdown Selection
    const dropdown = document.getElementById('case-select-dropdown');
    if (dropdown && dropdown.value !== String(caseId)) {
      dropdown.value = String(caseId);
    }

    try {
      // 1. Fetch Case Details
      const caseRes = await fetch(`/api/cases/${caseId}`);
      if (!caseRes.ok) throw new Error('Could not fetch case details');
      activeCaseData = await caseRes.json();
      renderCaseSummary(activeCaseData);

      // 2. Fetch Reconstructed Route Trajectory
      const routeRes = await fetch(`/api/route?case_id=${caseId}`);
      if (!routeRes.ok) throw new Error('Could not fetch route reconstruction');
      const routeData = await routeRes.json();

      // 3. Render Route Stats Strip
      renderRouteSummary(routeData);

      // 4. Plot Route on Map View
      if (window.MapView && typeof window.MapView.plotRoute === 'function') {
        window.MapView.plotRoute(routeData, (sightingId) => {
          if (window.TimelineView && typeof window.TimelineView.selectCard === 'function') {
            window.TimelineView.selectCard(sightingId);
          }
        });
      }

      // 5. Render Chronological Sighting Stream in Timeline View
      if (window.TimelineView && typeof window.TimelineView.renderTimeline === 'function') {
        window.TimelineView.renderTimeline(routeData.waypoints, caseId, onSightingVerified);
      }

      if (isRefresh) {
        showToast(`Route updated for ${activeCaseData.fir_number}`, 'info');
      }
    } catch (err) {
      console.error('Failed to load case:', err);
      showToast(`Error loading case: ${err.message}`, 'error');
    }
  }

  /**
   * Callback when an officer verifies or rejects a sighting
   */
  async function onSightingVerified(sightingId, status) {
    // Refresh route without full re-render
    try {
      const routeRes = await fetch(`/api/route?case_id=${activeCaseId}`);
      if (routeRes.ok) {
        const routeData = await routeRes.json();
        renderRouteSummary(routeData);
        if (window.MapView && typeof window.MapView.plotRoute === 'function') {
          window.MapView.plotRoute(routeData, (id) => {
            if (window.TimelineView) window.TimelineView.selectCard(id);
          });
        }
      }
      fetchDashboardStats();
    } catch (e) {
      console.warn('Silent route refresh error:', e);
    }
  }

  /**
   * Render Sidebar Active Case Summary
   */
  function renderCaseSummary(c) {
    const firEl = document.getElementById('case-fir-display');
    const plateEl = document.getElementById('case-plate-display');
    const statusEl = document.getElementById('case-status-badge');
    const vehicleEl = document.getElementById('case-vehicle-meta');
    const theftLocEl = document.getElementById('case-theft-location');
    const officerEl = document.getElementById('case-officer-meta');
    const featuresEl = document.getElementById('case-features-meta');

    if (firEl) firEl.innerText = c.fir_number;
    if (plateEl) plateEl.innerText = c.reported_plate;

    if (statusEl) {
      statusEl.className = `status-badge status-${(c.status || 'open').toLowerCase()}`;
      statusEl.innerText = c.status || 'OPEN';
    }

    if (vehicleEl) {
      vehicleEl.innerText = `${c.vehicle_color || ''} ${c.make || ''} ${c.model || ''} (${c.vehicle_type || 'Vehicle'})`;
    }

    if (theftLocEl) {
      theftLocEl.innerText = c.theft_location_name || 'Delhi NCR Region';
    }

    if (officerEl) {
      officerEl.innerText = c.investigating_officer || 'Inspector In-Charge';
    }

    if (featuresEl) {
      featuresEl.innerText = c.distinctive_features || 'Standard factory specifications';
    }
  }

  /**
   * Render Reconstructed Route Summary Strip
   */
  function renderRouteSummary(routeData) {
    const distEl = document.getElementById('route-stat-distance');
    const durEl = document.getElementById('route-stat-duration');
    const countEl = document.getElementById('route-stat-waypoints');
    const statusEl = document.getElementById('route-stat-status');

    if (distEl) distEl.innerText = `${routeData.total_distance_km || 0} km`;
    if (durEl) durEl.innerText = `${routeData.estimated_duration_min || 0} min`;
    if (countEl) countEl.innerText = `${routeData.waypoints ? routeData.waypoints.length : 0} nodes`;
    if (statusEl) {
      const hasInfeasible = (routeData.segments || []).some((s) => s.feasible === false);
      statusEl.innerText = hasInfeasible ? 'Speed Alert' : '100% Feasible';
      statusEl.style.color = hasInfeasible ? 'var(--crimson-red)' : 'var(--emerald-green)';
    }
  }

  /**
   * Bind All Interactive Controls & Modals
   */
  function bindEventHandlers() {
    // 1. Case Selector Dropdown
    const caseSelect = document.getElementById('case-select-dropdown');
    if (caseSelect) {
      caseSelect.addEventListener('change', (e) => {
        loadCase(Number(e.target.value));
      });
    }

    // 2. Refresh Route Button
    const refreshBtn = document.getElementById('btn-refresh-route');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => {
        if (activeCaseId) loadCase(activeCaseId, true);
      });
    }

    // 3. Export Section 65B Dossier Button
    const exportBtn = document.getElementById('btn-export-dossier');
    if (exportBtn) {
      exportBtn.addEventListener('click', () => {
        if (window.EvidenceExport && typeof window.EvidenceExport.openDossierModal === 'function') {
          window.EvidenceExport.openDossierModal(activeCaseId);
        }
      });
    }

    // 4. Quick Search Bar
    const quickSearchForm = document.getElementById('quick-search-form');
    const quickSearchInput = document.getElementById('quick-search-input');
    if (quickSearchForm && quickSearchInput) {
      quickSearchForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const query = quickSearchInput.value.trim().toUpperCase();
        if (!query) return;

        // Check if query matches a known FIR or Plate
        const match = allCases.find(
          (c) =>
            c.fir_number.toUpperCase().includes(query) ||
            c.reported_plate.toUpperCase().includes(query)
        );
        if (match) {
          loadCase(match.id);
          showToast(`Switched to matched case ${match.fir_number}`, 'success');
        } else {
          // Open Advanced Search pre-filled with query
          openSearchModalWithPlate(query);
        }
      });
    }

    // 5. Quick Filter Pills
    document.querySelectorAll('.filter-pill').forEach((pill) => {
      pill.addEventListener('click', () => {
        document.querySelectorAll('.filter-pill').forEach((p) => p.classList.remove('active'));
        pill.classList.add('active');
        const filterType = pill.dataset.filterType;
        const filterVal = pill.dataset.filterValue;
        if (filterType && filterVal) {
          openSearchModalWithFilters({ [filterType]: filterVal });
        }
      });
    });

    // 6. Modals Setup
    setupModalTriggers();

    // 7. New FIR Form Submission
    const firForm = document.getElementById('new-fir-form');
    if (firForm) {
      firForm.addEventListener('submit', handleNewFirSubmit);
    }

    // 8. Advanced Search Form Submission
    const searchForm = document.getElementById('advanced-search-form');
    if (searchForm) {
      searchForm.addEventListener('submit', handleAdvancedSearchSubmit);
    }

    // 9. Map Control Buttons
    const fitBoundsBtn = document.getElementById('btn-map-fit-bounds');
    if (fitBoundsBtn) {
      fitBoundsBtn.addEventListener('click', () => {
        if (window.MapView) window.MapView.fitRouteBounds();
      });
    }

    // Layer Toggles
    bindCheckbox('toggle-cameras', (checked) => window.MapView && window.MapView.toggleLayer('cameras', checked));
    bindCheckbox('toggle-waypoints', (checked) => window.MapView && window.MapView.toggleLayer('waypoints', checked));
    bindCheckbox('toggle-route', (checked) => window.MapView && window.MapView.toggleLayer('route', checked));

    // Basemap Switchers
    bindBasemapBtn('btn-base-dark', 'dark');
    bindBasemapBtn('btn-base-satellite', 'satellite');
    bindBasemapBtn('btn-base-streets', 'streets');
  }

  function bindCheckbox(id, callback) {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('change', (e) => callback(e.target.checked));
    }
  }

  function bindBasemapBtn(id, layerType) {
    const btn = document.getElementById(id);
    if (btn) {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.map-base-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        if (window.MapView) window.MapView.switchBaseLayer(layerType);
      });
    }
  }

  function setupModalTriggers() {
    // Open FIR Modal
    const newFirBtn = document.getElementById('btn-open-new-fir-modal');
    const firModal = document.getElementById('fir-modal');
    if (newFirBtn && firModal) {
      newFirBtn.addEventListener('click', () => {
        firModal.classList.add('active');
      });
    }

    // Open Search Modal
    const searchBtn = document.getElementById('btn-open-search-modal');
    const searchModal = document.getElementById('search-modal');
    if (searchBtn && searchModal) {
      searchBtn.addEventListener('click', () => {
        searchModal.classList.add('active');
      });
    }

    // Modal Close Buttons
    document.querySelectorAll('.modal-close-trigger').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.modal-backdrop').forEach((m) => m.classList.remove('active'));
      });
    });

    // Close on backdrop click
    document.querySelectorAll('.modal-backdrop').forEach((backdrop) => {
      backdrop.addEventListener('click', (e) => {
        if (e.target === backdrop) backdrop.classList.remove('active');
      });
    });

    // Reference Image Preview
    const fileInput = document.getElementById('fir-photo-input');
    const previewBox = document.getElementById('fir-photo-preview');
    if (fileInput && previewBox) {
      fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
          const reader = new FileReader();
          reader.onload = (re) => {
            previewBox.innerHTML = `<img src="${re.target.result}" style="max-height: 90px; border-radius: 4px;" />`;
            previewBox.dataset.base64 = re.target.result.split(',')[1];
          };
          reader.readAsDataURL(file);
        }
      });
    }
  }

  /**
   * Handle New FIR Registration Form Submission (POST /api/report)
   */
  async function handleNewFirSubmit(e) {
    e.preventDefault();

    const firNumber = document.getElementById('fir-number-input').value.trim();
    const reportedPlate = document.getElementById('fir-plate-input').value.trim().toUpperCase();
    const vehicleType = document.getElementById('fir-type-input').value;
    const vehicleColor = document.getElementById('fir-color-input').value;
    const make = document.getElementById('fir-make-input').value.trim();
    const model = document.getElementById('fir-model-input').value.trim();
    const features = document.getElementById('fir-features-input').value.trim();
    const theftTime = document.getElementById('fir-theft-time-input').value;
    const theftLocation = document.getElementById('fir-theft-location-input').value.trim();
    const theftLat = parseFloat(document.getElementById('fir-theft-lat-input').value) || 28.6315;
    const theftLon = parseFloat(document.getElementById('fir-theft-lon-input').value) || 77.2167;
    const officer = document.getElementById('fir-officer-input').value.trim() || 'Inspector Rajesh Kumar (DL-4821)';
    const station = document.getElementById('fir-station-input').value.trim() || 'Parliament Street Police Station';

    const previewBox = document.getElementById('fir-photo-preview');
    const refBase64 = previewBox ? previewBox.dataset.base64 : null;

    // Validate Indian License Plate Regex
    const cleanPlate = reportedPlate.replace(/[\s-]/g, '');
    const plateRegex = /^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$/;
    if (!plateRegex.test(cleanPlate)) {
      showToast(`Warning: '${reportedPlate}' deviates from standard HSRP format (e.g. MH12AB1234)`, 'error');
    }

    const payload = {
      fir_number: firNumber,
      reported_plate: cleanPlate,
      theft_datetime: theftTime ? new Date(theftTime).toISOString() : new Date().toISOString(),
      theft_latitude: theftLat,
      theft_longitude: theftLon,
      theft_location_name: theftLocation || 'Connaught Place, New Delhi',
      vehicle_type: vehicleType,
      vehicle_color: vehicleColor,
      make: make || 'Generic',
      model: model || 'Sedan',
      distinctive_features: features || 'None',
      reference_image_base64: refBase64,
      investigating_officer: officer,
      police_station: station
    };

    try {
      const res = await fetch('/api/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'FIR registration failed');
      }

      const newCase = await res.json();
      showToast(`FIR ${newCase.fir_number} Registered • ${newCase.total_candidate_sightings} Initial Matches Found!`, 'success');

      // Close modal and reset form
      document.getElementById('fir-modal').classList.remove('active');
      document.getElementById('new-fir-form').reset();
      if (previewBox) {
        previewBox.innerHTML = '<i class="fa-solid fa-cloud-arrow-up" style="font-size: 24px; color: #64748b;"></i><div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">Click or drag vehicle reference image</div>';
        delete previewBox.dataset.base64;
      }

      // Refresh cases list and load newly created case
      await fetchCases();
      await loadCase(newCase.id);
      fetchDashboardStats();
    } catch (err) {
      console.error('FIR registration error:', err);
      showToast(`Error: ${err.message}`, 'error');
    }
  }

  /**
   * Handle Advanced Multi-Modal Search (GET /api/search)
   */
  async function handleAdvancedSearchSubmit(e) {
    e.preventDefault();

    const plate = document.getElementById('search-plate-input').value.trim();
    const vType = document.getElementById('search-type-input').value;
    const color = document.getElementById('search-color-input').value;
    const cameraId = document.getElementById('search-camera-filter').value;
    const minConf = parseFloat(document.getElementById('search-conf-slider').value) / 100.0;
    const resultsContainer = document.getElementById('search-results-list');

    if (resultsContainer) {
      resultsContainer.innerHTML = '<div style="text-align: center; padding: 20px;"><i class="fa-solid fa-spinner fa-spin" style="font-size: 24px; color: #38bdf8;"></i><div style="font-size: 12px; margin-top: 8px;">Searching Camera Surveillance Sightings...</div></div>';
    }

    const params = new URLSearchParams();
    if (plate) params.append('plate', plate);
    if (vType) params.append('vehicle_type', vType);
    if (color) params.append('color', color);
    if (cameraId) params.append('camera_id', cameraId);
    params.append('min_confidence', minConf.toString());
    params.append('limit', '30');

    try {
      const res = await fetch(`/api/search?${params.toString()}`);
      if (!res.ok) throw new Error('Search request failed');
      const data = await res.json();
      renderSearchResults(data.results || [], resultsContainer);
    } catch (err) {
      console.error('Search error:', err);
      if (resultsContainer) {
        resultsContainer.innerHTML = `<div style="text-align: center; color: #ef4444; padding: 20px;">Search Error: ${err.message}</div>`;
      }
    }
  }

  function renderSearchResults(items, container) {
    if (!container) return;

    if (items.length === 0) {
      container.innerHTML = '<div style="text-align: center; padding: 30px; color: var(--text-muted);">No sightings found matching criteria.</div>';
      return;
    }

    container.innerHTML = '';
    const table = document.createElement('table');
    table.className = 'dossier-meta-table';
    table.style.fontSize = '11px';

    table.innerHTML = `
      <thead>
        <tr>
          <th>Time</th>
          <th>Camera Node</th>
          <th>Vehicle Crop</th>
          <th>Plate Crop</th>
          <th>Detected Plate</th>
          <th>OCR Conf</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody>
        ${items
          .map((s) => {
            const timeStr = s.timestamp
              ? new Date(s.timestamp).toLocaleTimeString('en-IN', { hour12: false })
              : 'N/A';
            const vCrop = s.crop_url || '/static/img/placeholder_car.jpg';
            const pCrop = s.plate_crop_url || '/static/img/placeholder_plate.jpg';
            return `
              <tr>
                <td style="font-family: monospace;">${timeStr}</td>
                <td><strong>${s.camera_id}</strong><br/><span style="font-size: 10px; color: #94a3b8;">${s.road_name || ''}</span></td>
                <td><img src="${vCrop}" style="width: 50px; height: 32px; object-fit: cover; border-radius: 2px;" onerror="this.style.display='none'" /></td>
                <td><img src="${pCrop}" style="width: 50px; height: 32px; object-fit: cover; border-radius: 2px;" onerror="this.style.display='none'" /></td>
                <td style="font-family: monospace; font-weight: bold; color: #fbbf24;">${s.plate_text || 'UNRESOLVED'}</td>
                <td style="font-weight: bold; color: #10b981;">${Math.round((s.plate_confidence || 0) * 100)}%</td>
                <td>
                  <button class="btn btn-primary btn-sm" onclick="App.highlightSightingOnMap(${s.camera_lat || 28.6315}, ${s.camera_lon || 77.2167}, '${s.plate_text || ''}')">
                    <i class="fa-solid fa-crosshairs"></i> View
                  </button>
                </td>
              </tr>
            `;
          })
          .join('')}
      </tbody>
    `;

    container.appendChild(table);
  }

  function highlightSightingOnMap(lat, lon, plateText) {
    document.getElementById('search-modal').classList.remove('active');
    if (window.MapView) {
      showToast(`Panning to sighting location for ${plateText}`, 'info');
    }
  }

  function openSearchModalWithPlate(plate) {
    const modal = document.getElementById('search-modal');
    const input = document.getElementById('search-plate-input');
    if (input) input.value = plate;
    if (modal) modal.classList.add('active');
    const form = document.getElementById('advanced-search-form');
    if (form) form.dispatchEvent(new Event('submit'));
  }

  function openSearchModalWithFilters(filters) {
    const modal = document.getElementById('search-modal');
    if (filters.type) {
      const typeInput = document.getElementById('search-type-input');
      if (typeInput) typeInput.value = filters.type;
    }
    if (filters.color) {
      const colorInput = document.getElementById('search-color-input');
      if (colorInput) colorInput.value = filters.color;
    }
    if (modal) modal.classList.add('active');
    const form = document.getElementById('advanced-search-form');
    if (form) form.dispatchEvent(new Event('submit'));
  }

  /**
   * Toast Notification Manager
   */
  function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    let icon = 'fa-info-circle';
    if (type === 'success') icon = 'fa-circle-check';
    if (type === 'error') icon = 'fa-triangle-exclamation';

    toast.innerHTML = `
      <i class="fa-solid ${icon}" style="font-size: 16px;"></i>
      <div style="flex: 1;">${message}</div>
    `;

    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(100%)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  function getOfficerBadgeId() {
    return officerBadgeId;
  }

  return {
    init,
    loadCase,
    showToast,
    getOfficerBadgeId,
    highlightSightingOnMap
  };
})();

// =============================================================================
// VIEW TAB SWITCHING & CAMERA GRID
// =============================================================================
function initViewTabs() {
  const tabs = document.querySelectorAll('.view-tab');
  const mapSection = document.getElementById('map-viewport-section');
  const cameraSection = document.getElementById('camera-grid-section');
  const workspace = document.querySelector('.main-workspace');

  if (!tabs.length || !mapSection || !cameraSection) return;

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      const view = tab.dataset.view;
      workspace.classList.remove('split-view');

      if (view === 'map') {
        mapSection.style.display = '';
        cameraSection.style.display = 'none';
      } else if (view === 'cameras') {
        mapSection.style.display = 'none';
        cameraSection.style.display = '';
      } else if (view === 'split') {
        mapSection.style.display = '';
        cameraSection.style.display = '';
        workspace.classList.add('split-view');
      }

      // Invalidate map size after layout change
      setTimeout(() => {
        if (window.MapView && window.MapView.initMap) {
          const map = window.MapView.initMap('map');
          if (map && map.invalidateSize) map.invalidateSize();
        }
      }, 200);
    });
  });
}

function populateCameraGrid(cameras) {
  const grid = document.getElementById('camera-grid');
  if (!grid || !cameras || !cameras.length) return;

  grid.innerHTML = '';

  cameras.forEach(cam => {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
    const dateStr = now.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' });

    const card = document.createElement('div');
    card.className = 'camera-feed-card';
    card.innerHTML = `
      <img class="camera-feed-img" id="cam-img-${cam.id}" src="/api/cameras/${cam.id}/frame?t=${Date.now()}" alt="${cam.name}" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22640%22 height=%22360%22><rect width=%22640%22 height=%22360%22 fill=%22%230a0e18%22/><text x=%2250%25%22 y=%2250%25%22 fill=%22%2364748b%22 text-anchor=%22middle%22 font-size=%2216%22 font-family=%22monospace%22>NO SIGNAL</text></svg>';" />
      <div class="camera-feed-hud">
        <div class="camera-hud-left">
          <div class="camera-hud-id"><i class="fa-solid fa-video"></i> ${cam.id}</div>
          <div class="camera-hud-name">${cam.name || cam.id}</div>
          <div class="camera-hud-road">${cam.road_name || ''}</div>
        </div>
        <div class="camera-hud-right">
          <div class="camera-hud-status online"><span class="live-pulse"></span> LIVE</div>
          <div class="camera-hud-time">${dateStr} ${timeStr} IST</div>
          <div class="camera-hud-rec">● REC 25 FPS</div>
        </div>
      </div>
      <div class="camera-feed-bottom">
        <div class="camera-gps-info">GPS: ${cam.latitude.toFixed(4)}°N, ${cam.longitude.toFixed(4)}°E | ${(cam.camera_type || 'urban').toUpperCase()}</div>
        <button class="anpr-scan-btn" data-cam-id="${cam.id}" title="Run Live ANPR Scan">
          <i class="fa-solid fa-crosshairs"></i> ANPR SCAN
        </button>
      </div>
      <div class="anpr-results-panel" id="anpr-results-${cam.id}" style="display:none;"></div>
    `;
    grid.appendChild(card);
  });

  // Bind ANPR scan buttons
  grid.querySelectorAll('.anpr-scan-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const camId = btn.dataset.camId;
      const panel = document.getElementById(`anpr-results-${camId}`);
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> SCANNING...';
      btn.disabled = true;

      try {
        const res = await fetch(`/api/anpr/scan-camera/${camId}`);
        if (!res.ok) throw new Error('Scan failed');
        const data = await res.json();

        if (data.detections && data.detections.length > 0) {
          let html = '<div class="anpr-results-header"><i class="fa-solid fa-id-card"></i> PLATES DETECTED</div>';
          data.detections.forEach(det => {
            html += `<div class="anpr-plate-result">
              <span class="anpr-plate-text">${det.plate_text}</span>
              <span class="anpr-plate-conf">${(det.confidence * 100).toFixed(0)}%</span>
            </div>`;
          });
          html += `<div class="anpr-engine-info">Engine: ${data.processing_engine} | SHA256: ${data.frame_hash_sha256.substring(0, 12)}...</div>`;
          panel.innerHTML = html;
          panel.style.display = 'block';

          btn.innerHTML = `<i class="fa-solid fa-check"></i> ${data.total_plates_found} PLATE(S)`;
          btn.style.background = 'rgba(16, 185, 129, 0.8)';
        } else {
          panel.innerHTML = '<div class="anpr-results-header">NO PLATES DETECTED</div>';
          panel.style.display = 'block';
          btn.innerHTML = '<i class="fa-solid fa-xmark"></i> NO PLATES';
          btn.style.background = 'rgba(239, 68, 68, 0.6)';
        }
      } catch (err) {
        btn.innerHTML = '<i class="fa-solid fa-exclamation-triangle"></i> ERROR';
        btn.style.background = 'rgba(239, 68, 68, 0.6)';
        console.error('ANPR scan error:', err);
      }

      setTimeout(() => {
        btn.innerHTML = '<i class="fa-solid fa-crosshairs"></i> ANPR SCAN';
        btn.disabled = false;
        btn.style.background = '';
      }, 5000);
    });
  });

  // Update timestamps every second
  setInterval(() => {
    const timeEls = grid.querySelectorAll('.camera-hud-time');
    const now = new Date();
    const ts = now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
    const ds = now.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' });
    timeEls.forEach(el => { el.textContent = `${ds} ${ts} IST`; });
  }, 1000);
}

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  App.init();
  initViewTabs();

  // Populate camera grid after cameras are fetched
  setTimeout(async () => {
    try {
      const res = await fetch('/api/cameras');
      if (res.ok) {
        const cameras = await res.json();
        populateCameraGrid(cameras);
      }
    } catch (e) {
      console.warn('Camera grid fetch error:', e);
    }
  }, 1500);
});

// Export globally
window.App = App;
