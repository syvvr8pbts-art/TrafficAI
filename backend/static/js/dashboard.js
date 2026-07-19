/*
 * dashboard.js — TrafficAI Command Center client
 *
 * The dashboard is a pure visualization of the backend /state payload. It does
 * NOT compute traffic values and contains NO random/placeholder data. The only
 * client-derived readouts are (a) live FPS + processing time from real frame_id
 * deltas, (b) connection state from real poll success/failure, and (c) an event
 * log built by observing real backend state transitions. Every other field is
 * taken verbatim from /state; missing values render as "--".
 *
 * The live video is a native MJPEG <img> pointed at /video_feed.
 */
(function () {
  "use strict";

  var POLL_MS = 1000; // unchanged — do not increase

  var LANE_ICONS = { North: "⬆️", East: "➡️", South: "⬇️", West: "⬅️" };
  // Vehicle cards. `stat` is the cumulative key in state.statistics, so this
  // panel shows session totals (unique vehicles), not per-frame visible counts.
  var COUNT_META = [
    { key: "car", stat: "cars", icon: "🚗", label: "Cars" },
    { key: "bike", stat: "bikes", icon: "🏍️", label: "Bikes" },
    { key: "auto-rickshaw", stat: "autos", icon: "🛺", label: "Auto" },
    { key: "cycle", stat: "cycles", icon: "🚲", label: "Cycle" },
    { key: "bus", stat: "buses", icon: "🚌", label: "Bus" },
    { key: "truck", stat: "trucks", icon: "🚚", label: "Truck" },
    { key: "ambulance", stat: "ambulances", icon: "🚑", label: "Ambulance" },
  ];
  // Cumulative statistics rows -> keys in state.statistics
  var STAT_ROWS = [
    { key: "frames", icon: "🎞️", label: "Frames Processed", head: false },
    { key: "vehicles", icon: "🚦", label: "Total Vehicles Processed", head: true },
    { key: "cars", icon: "🚗", label: "Cars" },
    { key: "bikes", icon: "🏍️", label: "Bikes" },
    { key: "autos", icon: "🛺", label: "Autos" },
    { key: "cycles", icon: "🚲", label: "Cycles" },
    { key: "buses", icon: "🚌", label: "Buses" },
    { key: "trucks", icon: "🚚", label: "Trucks" },
    { key: "ambulances", icon: "🚑", label: "Ambulances" },
  ];
  var DENSITY_FILL = { LOW: 34, MEDIUM: 67, HIGH: 100 };
  var MAX_EVENTS = 14;

  var $ = function (id) { return document.getElementById(id); };
  // Render a value or "--" when unavailable (never fabricate).
  function val(v, suffix) {
    if (v === null || v === undefined || v === "") return "--";
    return suffix ? v + suffix : String(v);
  }

  var els = {
    body: document.body,
    loader: $("loader"),
    connBanner: $("connBanner"),
    connChip: $("connChip"),
    connText: $("connText"),
    clockTime: $("clockTime"),
    clockDate: $("clockDate"),
    emergencyBar: $("emergencyBar"),
    emergencyLane: $("emergencyLane"),
    fpsBadge: $("fpsBadge"),
    frameBadge: $("frameBadge"),
    trafficLight: document.querySelector(".traffic-light"),
    countdown: $("countdown"),
    countdownRing: $("countdownRing"),
    curLane: $("curLane"),
    curSignal: $("curSignal"),
    nextLane: $("nextLane"),
    laneGrid: $("laneGrid"),
    densityBadge: $("densityBadge"),
    densityBar: $("densityBar"),
    occupancy: $("occupancy"),
    occupancyRing: $("occupancyRing"),
    recGreen: $("recGreen"),
    aiDecision: $("aiDecision"),
    intersection: $("intersection"),
    emergency: $("emergency"),
    totalBadge: $("totalBadge"),
    countsGrid: $("countsGrid"),
    statsList: $("statsList"),
    eventLog: $("eventLog"),
    hBackend: $("hBackend"),
    hConnection: $("hConnection"),
    hFps: $("hFps"),
    hProc: $("hProc"),
    hFrame: $("hFrame"),
    hStream: $("hStream"),
  };

  // presentation-only client state
  var booted = false;
  var lastCounts = {};
  var phaseKey = null, phaseTotal = 0;
  var fps = { lastFrame: null, lastTime: null, value: 0 };
  var prevState = null;
  var events = [];
  var ringCirc = {};

  function bulbFor(signal) {
    if (signal === "GREEN") return "green";
    if (signal === "YELLOW") return "yellow";
    return "red";
  }
  function ringColorClass(signal) {
    if (signal === "GREEN") return "ring--green";
    if (signal === "YELLOW") return "ring--yellow";
    return "ring--red";
  }
  function setRing(circle, frac) {
    if (!circle) return;
    var r = parseFloat(circle.getAttribute("r"));
    var c = ringCirc[circle.id] || (ringCirc[circle.id] = 2 * Math.PI * r);
    circle.style.strokeDasharray = c.toFixed(1);
    circle.style.strokeDashoffset = (c * (1 - Math.max(0, Math.min(1, frac)))).toFixed(1);
  }
  function show(el) { el.classList.remove("hidden"); }
  function hide(el) { el.classList.add("hidden"); }

  // ---- connection ----
  function setConnection(up) {
    if (up) {
      els.connChip.className = "status-chip status-chip--up";
      els.connText.textContent = "Backend Online";
      hide(els.connBanner);
    } else {
      els.connChip.className = "status-chip status-chip--lost";
      els.connText.textContent = "Connection Lost";
      show(els.connBanner);
    }
    els.hBackend.textContent = up ? "Online" : "Offline";
    els.hBackend.className = "health__value " + (up ? "ok" : "bad");
    els.hConnection.textContent = up ? "Live" : "Lost";
    els.hConnection.className = "health__value " + (up ? "ok" : "bad");
  }

  // ---- FPS / processing (derived from real frame_id advance) ----
  function updateFps(frameId) {
    var now = performance.now();
    if (fps.lastFrame != null && frameId != null && now > fps.lastTime) {
      var df = frameId - fps.lastFrame, dt = (now - fps.lastTime) / 1000;
      if (df >= 0 && dt > 0) {
        var inst = df / dt;
        fps.value = fps.value ? fps.value * 0.6 + inst * 0.4 : inst;
      }
    }
    fps.lastFrame = frameId;
    fps.lastTime = now;
    var hasFps = fps.value > 0;
    els.fpsBadge.textContent = (hasFps ? fps.value.toFixed(0) : "--") + " FPS";
    els.hFps.textContent = hasFps ? fps.value.toFixed(1) : "--";
    els.hProc.textContent = hasFps ? (1000 / fps.value).toFixed(0) + " ms" : "--";
    els.hStream.textContent = hasFps ? "Active" : "Idle";
    els.hStream.className = "health__value " + (hasFps ? "ok" : "");
  }

  // ---- event log (from real backend transitions) ----
  function pushEvent(text, type) {
    var d = new Date();
    events.unshift({ t: d.toLocaleTimeString([], { hour12: false }), text: text, type: type || "info" });
    if (events.length > MAX_EVENTS) events.pop();
    els.eventLog.innerHTML = events.map(function (e) {
      return '<li class="event event--' + e.type + '">' +
        '<span class="event__time">' + e.t + '</span>' +
        '<span class="event__text">' + e.text + "</span></li>";
    }).join("");
  }
  function diffEvents(s) {
    var p = prevState;
    if (!p) { pushEvent("Backend connected · monitoring live", "ok"); return; }
    if (s.ambulance === true && p.ambulance !== true)
      pushEvent("🚑 Emergency priority activated" + (s.ambulance_lane ? " · " + s.ambulance_lane : ""), "emergency");
    if (s.ambulance !== true && p.ambulance === true)
      pushEvent("Emergency cleared · normal operation resumed", "ok");
    if (s.lane !== p.lane && s.lane)
      pushEvent("Lane switched " + (p.lane || "--") + " → " + s.lane, "info");
    else if (s.signal !== p.signal && s.signal)
      pushEvent("Signal → " + s.signal + (s.lane ? " (" + s.lane + ")" : ""), "info");
    if (s.intersection !== p.intersection && s.intersection)
      pushEvent("Intersection " + s.intersection, s.intersection === "OCCUPIED" ? "warn" : "ok");
  }

  // ---- renderers ----
  function renderSignal(s) {
    var signal = s.signal;
    var active = bulbFor(signal);
    els.trafficLight.querySelectorAll(".bulb").forEach(function (b) {
      b.classList.toggle("is-on", signal ? b.dataset.bulb === active : false);
    });
    els.curLane.textContent = val(s.lane);
    els.curSignal.textContent = val(signal);
    els.nextLane.textContent = val(s.next_lane);

    var cd = s.countdown;
    els.countdown.textContent = val(cd);
    if (cd != null) {
      var key = (s.lane || "") + "|" + (signal || "");
      if (key !== phaseKey || cd > phaseTotal) { phaseKey = key; phaseTotal = Math.max(cd, 1); }
      els.countdownRing.setAttribute("class", "ring__value " + ringColorClass(signal));
      setRing(els.countdownRing, phaseTotal ? cd / phaseTotal : 0);
    }
  }

  function renderLanes(s) {
    var lanes = Array.isArray(s.lanes) && s.lanes.length ? s.lanes : null;
    if (!lanes) return; // keep skeletons if backend hasn't reported lanes
    els.laneGrid.innerHTML = lanes.map(function (lane) {
      var isActive = lane === s.lane;
      var dot = isActive ? bulbFor(s.signal) : "";
      var stateClass = isActive && s.signal ? "state-" + s.signal : "";
      return '<div class="lane ' + (isActive ? "is-active " : "") + stateClass + '">' +
        '<span class="lane__name"><span class="lane__icon">' + (LANE_ICONS[lane] || "") + "</span>" + lane + "</span>" +
        '<span class="lane__dot ' + dot + '"></span></div>';
    }).join("");
  }

  function renderMetrics(s) {
    var d = (s.density || "").toUpperCase();
    var tone = d === "HIGH" ? "high" : d === "MEDIUM" ? "medium" : d === "LOW" ? "low" : "";
    els.densityBar.className = "bar__fill " + tone;
    els.densityBar.style.width = (DENSITY_FILL[d] || 0) + "%";
    els.densityBadge.className = "chip " + (tone ? "chip--" + tone : "chip--muted");
    els.densityBadge.textContent = val(s.density);

    var occ = typeof s.occupancy === "number" ? s.occupancy : null;
    els.occupancy.textContent = occ != null ? Math.round(occ) + "%" : "--";
    setRing(els.occupancyRing, occ != null ? occ / 100 : 0);

    els.recGreen.textContent = val(s.recommended_green);
  }

  function renderAiDecision(s) {
    if (s.ambulance === true) {
      els.aiDecision.textContent = "Emergency corridor — holding GREEN for " + val(s.ambulance_lane || s.lane) + " until the ambulance clears the intersection.";
      return;
    }
    if (!s.lane || !s.signal) { els.aiDecision.textContent = "--"; return; }
    if (s.signal === "GREEN") {
      els.aiDecision.textContent = "Serving " + s.lane + " with GREEN for " + val(s.recommended_green) +
        "s based on " + val(s.density) + " traffic density.";
    } else {
      els.aiDecision.textContent = s.lane + " in " + s.signal + " phase · preparing next lane " + val(s.next_lane) + ".";
    }
  }

  function renderStatus(s) {
    var iel = els.intersection;
    if (s.intersection === "CLEAR") { iel.textContent = "✔ CLEAR"; iel.className = "status-badge status-badge--green"; }
    else if (s.intersection === "OCCUPIED") { iel.textContent = "✖ OCCUPIED"; iel.className = "status-badge status-badge--red"; }
    else { iel.textContent = "--"; iel.className = "status-badge status-badge--muted"; }

    var eel = els.emergency;
    if (s.ambulance === true) {
      eel.textContent = "🚑 ACTIVE" + (s.ambulance_lane ? " · " + s.ambulance_lane : "");
      eel.className = "status-badge status-badge--amber";
    } else if (s.ambulance === false) {
      eel.textContent = "Normal Operation";
      eel.className = "status-badge status-badge--green";
    } else {
      eel.textContent = "--";
      eel.className = "status-badge status-badge--muted";
    }

    // Emergency banner — gated STRICTLY on backend ambulance === true.
    if (s.ambulance === true) {
      show(els.emergencyBar);
      els.emergencyLane.textContent = s.ambulance_lane ? "Lane · " + s.ambulance_lane : "";
    } else {
      hide(els.emergencyBar);
    }
  }

  function renderCounts(s) {
    // Cumulative session totals (unique vehicles) — never decrease.
    var stats = s.statistics || {};
    var hasStats = Object.keys(stats).length > 0;
    els.totalBadge.textContent = (typeof stats.vehicles === "number" ? stats.vehicles : "--") + " total";
    els.countsGrid.innerHTML = COUNT_META.map(function (m) {
      var has = hasStats && stats[m.stat] != null;
      var v = has ? stats[m.stat] : "--";
      var alert = m.key === "ambulance" && has && stats[m.stat] > 0 ? " count--alert" : "";
      var bump = has && lastCounts[m.stat] != null && stats[m.stat] !== lastCounts[m.stat] ? " bump" : "";
      return '<div class="count' + alert + bump + '">' +
        '<div class="count__icon">' + m.icon + "</div>" +
        '<div class="count__value">' + v + "</div>" +
        '<div class="count__label">' + m.label + "</div></div>";
    }).join("");
    if (hasStats) lastCounts = Object.assign({}, stats);
  }

  function renderStats(s) {
    var st = s.statistics || {};
    els.statsList.innerHTML = STAT_ROWS.map(function (r) {
      var v = st[r.key] != null ? st[r.key] : "--";
      return '<div class="stat-row' + (r.head ? " stat-row--head" : "") + '">' +
        '<span class="stat-row__label">' + r.icon + " " + r.label + "</span>" +
        '<span class="stat-row__value">' + v + "</span></div>";
    }).join("");
  }

  function render(s) {
    updateFps(s.frame_id);
    els.frameBadge.textContent = "frame " + val(s.frame_id);
    els.hFrame.textContent = val(s.frame_id);
    renderSignal(s);
    renderLanes(s);
    renderMetrics(s);
    renderAiDecision(s);
    renderStatus(s);
    renderCounts(s);
    renderStats(s);
    diffEvents(s);
    prevState = s;
  }

  // ---- polling ----
  function poll() {
    fetch("/state", { cache: "no-store" })
      .then(function (res) { if (!res.ok) throw new Error("HTTP " + res.status); return res.json(); })
      .then(function (s) {
        setConnection(true);
        if (!booted) { booted = true; els.body.classList.remove("is-booting"); els.loader.classList.add("is-hidden"); }
        render(s);
      })
      .catch(function () {
        setConnection(false); // keeps last real values; never fabricates
      });
  }

  function tickClock() {
    var now = new Date();
    els.clockTime.textContent = now.toLocaleTimeString([], { hour12: false });
    els.clockDate.textContent = now.toLocaleDateString([], { weekday: "short", year: "numeric", month: "short", day: "numeric" });
  }

  tickClock();
  setInterval(tickClock, 1000);
  setInterval(poll, POLL_MS);
  poll();
})();
