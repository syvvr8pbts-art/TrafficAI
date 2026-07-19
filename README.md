# 🚦 TrafficAI — AI-Based Intelligent Traffic Management System

Real-time, camera-driven traffic control. TrafficAI detects vehicles with YOLO,
derives live analytics (counts, density, occupancy), computes **adaptive signal
timing**, runs a safe traffic-light **state machine** with **ambulance priority**
and **intersection clearance**, and streams everything to a professional
**command-center dashboard**.

---

## ✨ Features

- **YOLO vehicle detection** (Ultralytics) — car, bike, cycle, auto-rickshaw, bus, truck, ambulance
- **Traffic analytics** — per-class counts, density (LOW/MEDIUM/HIGH), lane occupancy, cumulative statistics, CSV logging
- **Adaptive signal intelligence** — demand-based green-time recommendation, GREEN → YELLOW → ALL_RED → CLEARANCE state machine (never two greens at once)
- **Ambulance priority** — emergency green corridor while an ambulance is present, then safe resume
- **Intersection clearance** — occupancy-gated phase change (waits until the box is physically clear)
- **Live dashboard** — MJPEG video, animated traffic light, circular countdown, lane cards, analytics, emergency banner; polls once per second
- **Clean modular backend** — each stage is a single-responsibility consumer on an `AnalyticsPipeline`

---

## 🧱 Architecture

```
                 Traffic Video (file / camera / RTSP)
                              │
                        OpenCV capture
                              │
                     YOLO Detection (detector.py)
                              │
              ┌──────────────  AnalyticsPipeline  ──────────────┐
              │  Counter → Density → Occupancy → Statistics →   │
              │  Recommendation → Signal Controller →           │
              │  Clearance → Ambulance → Logger                 │
              └───────────────────────┬─────────────────────────┘
                                      │  (each stage appends to one payload)
                              TrafficSystem (state)
                              │                  │
                     Flask API (api.py)   VideoProcessor (streamer.py)
                     /state /signal              │  annotated MJPEG
                     /health                     │
                              │                  │
                     ┌────────┴──────────────────┴────────┐
                     │   Dashboard (HTML / CSS / JS)       │
                     │   video + signal + analytics live   │
                     └─────────────────────────────────────┘
```

Data flows one way: the detector emits a structured payload, each analytics/signal
module **appends** its own fields, and the dashboard only **renders** them (it
computes nothing).

---

## 📂 Project Structure

```
TrafficAI/
├── backend/                     # Serving layer (Flask + pipeline glue)
│   ├── detector.py              # Frame → structured detection payload (wraps YOLO)
│   ├── pipeline.py              # AnalyticsPipeline + DetectionConsumer contract
│   ├── counter.py               # Per-class vehicle counts
│   ├── density.py               # LOW / MEDIUM / HIGH density
│   ├── occupancy.py             # % frame area covered by vehicles
│   ├── statistics.py            # Cumulative session totals
│   ├── recommendation.py        # Adaptive green-time recommendation
│   ├── signal_controller.py     # Traffic-light finite state machine
│   ├── clearance.py             # Intersection ROI occupancy gating
│   ├── ambulance.py             # Emergency green corridor
│   ├── logger.py                # Per-frame CSV log
│   ├── streamer.py              # Background video processor + MJPEG
│   ├── app.py                   # Composition root / entry point / TrafficSystem
│   ├── api.py                   # Flask routes
│   ├── config.py                # Backend/server config (reuses root config)
│   ├── utils.py                 # Helpers (logging, timestamps)
│   ├── templates/dashboard.html # Dashboard markup
│   └── static/
│       ├── css/dashboard.css    # Dashboard styling
│       └── js/dashboard.js      # Dashboard client (1s polling)
├── config.py / detector.py / tracker.py / ...   # Original standalone pipeline (main.py)
├── weights/                     # YOLO weights (e.g. yolo11s.pt or best.pt)
├── videos/                      # Input videos (e.g. sample.mp4)
├── logs/                        # Auto-generated detection CSVs
├── output/                      # Legacy pipeline outputs
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

- Python 3.11
- See `requirements.txt`: `ultralytics`, `opencv-python`, `numpy`, `scipy`, `shapely`, `supervision`, `flask`
- A YOLO weights file in `weights/` (ships/works with `yolo11s.pt`; drop your trained `best.pt` there to use custom classes)

---

## 🚀 Installation

```bash
# 1. clone / enter the project
cd TrafficAI

# 2. create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4. ensure a weights file exists
#    weights/yolo11s.pt  (works out of the box)
#    weights/best.pt     (your custom-trained model, optional)

# 5. put a demo video at videos/sample.mp4
```

Device is auto-selected (CUDA → Apple MPS → CPU); no configuration needed.

---

## ▶️ Run Commands

**Dashboard (main demo):**
```bash
python -m backend.app --serve --source videos/sample.mp4
# open http://127.0.0.1:8000/dashboard
```

**Headless analytics (terminal + CSV):**
```bash
python -m backend.app --analyze videos/sample.mp4
python -m backend.app --analyze videos/sample.mp4 --max-frames 200
```

**Payload contract self-test (no model load):**
```bash
python -m backend.app --selftest
```

**Original standalone pipeline (annotated video + JSON/CSV report):**
```bash
python main.py --source videos/sample.mp4
```

### API endpoints
| Endpoint | Description |
|---|---|
| `GET /dashboard` | The live dashboard page |
| `GET /video_feed` | Processed MJPEG stream |
| `GET /state` | Full live state (signal + analytics) — dashboard source |
| `GET /signal` | Compact signal state |
| `GET /health` | Liveness probe |

---

## 🎬 Demo Commands

```bash
# start the control center
python -m backend.app --serve --source videos/sample.mp4
```
Then open **http://127.0.0.1:8000/dashboard** on the projector. To showcase
ambulance priority, use a clip containing an ambulance (with custom `best.pt`).

---

## ⚠️ Known Limitations

- **Single intersection** per running instance (the architecture supports more; one is instantiated for the demo).
- **Ambulance / auto-rickshaw** require the custom-trained `best.pt`; the stock COCO `yolo11s.pt` does not emit those classes.
- **Signal timing runs on wall-clock seconds** during processing (great for a live demo; not synced to source-video timestamps).
- **Occupancy** is an area-coverage approximation (overlapping boxes are not de-duplicated).
- Ambulance **lane direction** uses a configurable default lane unless lane ROIs are provided.
- Uses the **Flask development server** (fine for a demo; use gunicorn/uvicorn behind a reverse proxy for production).

---

## 🔭 Future Scope

- Multi-intersection coordination and "green wave" corridors
- Fine-tuned detector for regional vehicle classes + higher ambulance recall
- WebSocket push (replace polling) and hardware signal integration (GPIO / controller protocols)
- Historical dashboards and analytics database
- Reinforcement-learning signal policy (drop-in `DecisionPolicy`)

---

## 🧪 Quality

- Modular, single-responsibility backend with a one-way data flow
- Thread-safe state sharing between the video processor and the API
- Graceful handling of missing/invalid video sources and camera disconnects
- Verified end-to-end: detection, analytics, signal FSM, ambulance override, clearance gating, dashboard, and all API endpoints
