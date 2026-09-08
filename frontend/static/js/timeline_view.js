/**
 * TimelineView - Chronological Sighting Stream & Officer Verification Controller
 * Renders high-resolution vehicle and license plate crops, multi-metric confidence badges,
 * Section 65B cryptographic digests, and officer Confirm / Reject action controls.
 */

const TimelineView = (function () {
  let activeCaseId = null;
  let onVerifySuccessCallback = null;
  let selectedCardId = null;

  /**
   * Helper: Copy text to clipboard and trigger notification
   */
  function copyToClipboard(text, label = 'SHA-256 Hash') {
    navigator.clipboard.writeText(text).then(
      () => {
        if (window.App && typeof window.App.showToast === 'function') {
          window.App.showToast(`${label} copied to clipboard!`, 'info');
        }
      },
      (err) => {
        console.error('Could not copy text: ', err);
      }
    );
  }

  /**
   * Render chronological sighting cards into sidebar container
   */
  function renderTimeline(waypoints, caseId, onVerifyCallback) {
    activeCaseId = caseId;
    onVerifySuccessCallback = onVerifyCallback;

    const container = document.getElementById('timeline-container');
    if (!container) return;

    container.innerHTML = '';

    if (!waypoints || waypoints.length === 0) {
      container.innerHTML = `
        <div style="text-align: center; padding: 40px 20px; color: var(--text-muted);">
          <i class="fa-solid fa-satellite-dish" style="font-size: 32px; margin-bottom: 12px; color: #334155;"></i>
          <div style="font-size: 14px; font-weight: 600; color: var(--text-secondary); margin-bottom: 6px;">
            No Candidate Sightings Tracked
          </div>
          <div style="font-size: 11px;">
            No automated ANPR or Re-ID matches currently linked to this FIR case. Use Advanced Search to associate new camera detections.
          </div>
        </div>
      `;
      return;
    }

    const trackWrapper = document.createElement('div');
    trackWrapper.className = 'timeline-track-wrapper';

    waypoints.forEach((wp, idx) => {
      const card = createSightingCard(wp, idx + 1);
      trackWrapper.appendChild(card);
    });

    container.appendChild(trackWrapper);
  }

  /**
   * Create an individual sighting DOM card element
   */
  function createSightingCard(wp, sequenceNum) {
    const card = document.createElement('div');
    card.className = `sighting-card ${wp.verification_status || 'pending'}`;
    card.id = `sighting-card-${wp.sighting_id}`;
    card.dataset.sightingId = wp.sighting_id;

    // Sighting Pin
    const pin = document.createElement('div');
    pin.className = 'sighting-node-pin';
    pin.innerText = `#${wp.sequence || sequenceNum}`;
    card.appendChild(pin);

    // Card Inner Container
    const inner = document.createElement('div');
    inner.className = 'sighting-card-inner';

    // Format Timestamp
    const formattedDate = wp.timestamp
      ? new Date(wp.timestamp).toLocaleString('en-IN', {
          month: 'short',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit'
        })
      : 'N/A';

    // Image URLs with resilient fallback
    const vehicleCropUrl = wp.vehicle_crop_url || '/static/img/placeholder_car.jpg';
    const plateCropUrl = wp.plate_crop_url || '/static/img/placeholder_plate.jpg';

    // Confidence / Scores
    const plateConfidencePct = Math.round((wp.plate_confidence || 0) * 100);
    const compositeScorePct = Math.round((wp.composite_score || 0.9) * 100);
    const speedKmh = wp.speed_from_prev_kmh;
    const isFeasible = wp.is_spatially_feasible !== false;

    // SHA-256 Digest handling
    const fullHash = wp.sha256_hash && wp.sha256_hash !== 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855' ? wp.sha256_hash : null;
    const truncHash = fullHash ? `${fullHash.substring(0, 10)}...${fullHash.substring(fullHash.length - 8)}` : 'NO DIGEST';

    inner.innerHTML = `
      <!-- Header -->
      <div class="sighting-card-header">
        <div class="sighting-camera-title">
          <span class="cam-name"><i class="fa-solid fa-video" style="color: #38bdf8; font-size: 11px;"></i> ${wp.camera_name || wp.camera_id}</span>
          <span class="cam-loc">${wp.road_name || 'Corridor Node'}</span>
        </div>
        <span class="sighting-timestamp-pill">${formattedDate}</span>
      </div>

      <!-- Media Grid -->
      <div class="sighting-media-grid">
        <div class="media-thumbnail-box" title="Vehicle Full Crop">
          <span class="media-tag">Vehicle</span>
          <img src="${vehicleCropUrl}" alt="Vehicle Crop" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'120\\' height=\\'80\\' fill=\\'%231e293b\\'><rect width=\\'120\\' height=\\'80\\'/><text x=\\'50%\\' y=\\'50%\\' fill=\\'%2364748b\\' text-anchor=\\'middle\\' font-size=\\'11\\'>Vehicle Crop</text></svg>';" />
        </div>
        <div class="media-thumbnail-box" title="License Plate Localization Zoom">
          <span class="media-tag">Plate Zoom</span>
          <img src="${plateCropUrl}" alt="Plate Crop" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'120\\' height=\\'80\\' fill=\\'%231e293b\\'><rect width=\\'120\\' height=\\'80\\'/><text x=\\'50%\\' y=\\'50%\\' fill=\\'%2364748b\\' text-anchor=\\'middle\\' font-size=\\'11\\'>Plate Zoom</text></svg>';" />
          <div class="plate-ocr-overlay">${wp.plate_text || 'MH12AB1234'}</div>
        </div>
      </div>

      <!-- Sighting Confidence & Spatio-Temporal Metrics -->
      <div class="sighting-metrics-row">
        <span class="metric-chip chip-conf-high" title="License Plate OCR Confidence">
          <i class="fa-solid fa-font"></i> OCR: ${plateConfidencePct}%
        </span>
        <span class="metric-chip chip-reid" title="Deep Metric Visual Re-ID Match">
          <i class="fa-solid fa-fingerprint"></i> Match: ${compositeScorePct}%
        </span>
        ${
          speedKmh !== null && speedKmh !== undefined
            ? `<span class="metric-chip ${isFeasible ? 'chip-speed' : 'chip-infeasible'}" title="Speed from Previous Camera">
                <i class="fa-solid fa-gauge-high"></i> ${speedKmh} km/h ${isFeasible ? '' : '(OVERLIMIT)'}
              </span>`
            : `<span class="metric-chip chip-speed"><i class="fa-solid fa-flag"></i> Origin Sighting</span>`
        }
      </div>

      <!-- Section 65B Hash Chip -->
      <div class="sighting-sha-chip" title="Section 65B Bit-Exact SHA-256 Forensic Digest">
        <span><strong style="color: #64748b;">SHA-256:</strong> ${truncHash}</span>
        ${
          fullHash
            ? `<button class="copy-hash-btn" title="Copy Complete SHA-256 Digest" data-hash="${fullHash}">
                <i class="fa-regular fa-copy"></i> Copy
              </button>`
            : ''
        }
      </div>

      <!-- Officer Verification Actions -->
      <div class="sighting-verify-action-bar" id="verify-bar-${wp.sighting_id}">
        ${renderVerificationControls(wp.sighting_id, wp.verification_status)}
      </div>
    `;

    // Bind Copy Button
    const copyBtn = inner.querySelector('.copy-hash-btn');
    if (copyBtn) {
      copyBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        copyToClipboard(copyBtn.dataset.hash, `Sighting #${wp.sequence} SHA-256 Hash`);
      });
    }

    // Card Click Selection
    inner.addEventListener('click', () => {
      selectCard(wp.sighting_id);
      if (window.MapView && typeof window.MapView.highlightWaypoint === 'function') {
        window.MapView.highlightWaypoint(wp.sighting_id);
      }
    });

    // Bind Verification Buttons
    bindVerificationButtons(inner, wp.sighting_id);

    card.appendChild(inner);
    return card;
  }

  /**
   * Helper to render buttons or verified label
   */
  function renderVerificationControls(sightingId, status) {
    if (status === 'verified') {
      return `
        <span class="verification-status-label verified">
          <i class="fa-solid fa-circle-check"></i> VERIFIED MATCH
        </span>
        <div class="verify-btn-group">
          <button class="btn-verify-reject btn-sm" data-sighting-id="${sightingId}" title="Reject Match">
            <i class="fa-solid fa-xmark"></i> Reject
          </button>
        </div>
      `;
    } else if (status === 'rejected') {
      return `
        <span class="verification-status-label rejected">
          <i class="fa-solid fa-circle-xmark"></i> REJECTED
        </span>
        <div class="verify-btn-group">
          <button class="btn-verify-confirm btn-sm" data-sighting-id="${sightingId}" title="Verify Match">
            <i class="fa-solid fa-check"></i> Verify
          </button>
        </div>
      `;
    } else {
      return `
        <span class="verification-status-label pending">
          <i class="fa-regular fa-clock"></i> PENDING REVIEW
        </span>
        <div class="verify-btn-group">
          <button class="btn-verify-confirm btn-sm" data-sighting-id="${sightingId}" title="Confirm Vehicle Match">
            <i class="fa-solid fa-check"></i> Confirm
          </button>
          <button class="btn-verify-reject btn-sm" data-sighting-id="${sightingId}" title="Reject False Positive">
            <i class="fa-solid fa-xmark"></i> Reject
          </button>
        </div>
      `;
    }
  }

  /**
   * Bind event listeners for verify and reject buttons
   */
  function bindVerificationButtons(container, sightingId) {
    const confirmBtn = container.querySelector('.btn-verify-confirm');
    if (confirmBtn) {
      confirmBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        executeVerification(sightingId, 'verified');
      });
    }

    const rejectBtn = container.querySelector('.btn-verify-reject');
    if (rejectBtn) {
      rejectBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        executeVerification(sightingId, 'rejected');
      });
    }
  }

  /**
   * Send POST /api/verify to FastAPI backend
   */
  async function executeVerification(sightingId, status) {
    if (!activeCaseId) return;

    const officerBadgeId = (window.App && window.App.getOfficerBadgeId()) || 'DL-4821';
    const payload = {
      case_id: activeCaseId,
      sighting_id: sightingId,
      status: status,
      officer_badge_id: officerBadgeId,
      notes: `Officer in-situ command dashboard verification: ${status}`
    };

    try {
      const res = await fetch('/api/verify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Verification request failed');
      }

      const data = await res.json();

      // Update Card DOM
      const card = document.getElementById(`sighting-card-${sightingId}`);
      if (card) {
        card.className = `sighting-card ${status}`;
        const verifyBar = card.querySelector(`#verify-bar-${sightingId}`);
        if (verifyBar) {
          verifyBar.innerHTML = renderVerificationControls(sightingId, status);
          bindVerificationButtons(verifyBar, sightingId);
        }
      }

      if (window.App && typeof window.App.showToast === 'function') {
        window.App.showToast(
          `Sighting #${sightingId} marked as ${status.toUpperCase()} by Badge #${officerBadgeId}`,
          status === 'verified' ? 'success' : 'error'
        );
      }

      // Notify parent callback
      if (typeof onVerifySuccessCallback === 'function') {
        onVerifySuccessCallback(sightingId, status);
      }
    } catch (err) {
      console.error('Verification error:', err);
      if (window.App && typeof window.App.showToast === 'function') {
        window.App.showToast(`Verification Failed: ${err.message}`, 'error');
      }
    }
  }

  /**
   * Select a sighting card visually
   */
  function selectCard(sightingId) {
    selectedCardId = sightingId;
    document.querySelectorAll('.sighting-card').forEach((c) => c.classList.remove('selected'));
    const target = document.getElementById(`sighting-card-${sightingId}`);
    if (target) {
      target.classList.add('selected');
      target.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  return {
    renderTimeline,
    selectCard
  };
})();

// Export globally
window.TimelineView = TimelineView;
