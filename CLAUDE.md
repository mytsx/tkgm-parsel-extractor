# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TKGM Parsel Extractor - A desktop application that bulk-extracts parcel data from the Turkish Land Registry (TKGM) API for a defined geographic area. Uses a Cloudflare Workers proxy to bypass rate limits.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the desktop application
python app.py

# Run CLI client (for scripting)
python tkgm_client.py

# Build standalone executable
pip install pyinstaller
python build.py
# Output: dist/TKGM-Parsel.app (macOS) or dist/TKGM-Parsel.exe (Windows)
```

## Architecture

### Component Overview

1. **app.py** - PyQt5 desktop application with Fluent Design UI (qfluentwidgets)
   - `KMLParser`: Parses KML files, extracts polygons, implements point-in-polygon
   - `TKGMClient`: HTTP client with single (`/parsel`) and batch (`/batch`) requests
   - `ScanWorker`: QThread that implements the scanning algorithm with batch + pruning
   - `MainWindow`: UI with worker URL config, KML loading, scan controls, export

2. **tkgm_client.py** - Standalone CLI client for scripting/programmatic use
   - Same `TKGMClient` class but standalone
   - Can use direct TKGM API (rate limited) or Worker proxy

3. **cloudflare-worker.js** - Cloudflare Workers proxy
   - `GET /parsel/{lat}/{lon}` - Single coordinate query
   - `POST /batch` - Batch query (up to 20 coordinates, staggered parallel requests)
   - Optional API Key protection via `X-API-Key` header and `API_KEY` secret

### Scanning Algorithm (app.py ScanWorker)

The scanning uses a **Batch + Pruning hybrid** approach:
1. Generate grid points within the KML polygon(s)
2. Send coordinates in batches of 15 to `/batch` endpoint
3. For each found parcel, **prune** remaining points that fall inside its geometry
4. This eliminates redundant queries for points already covered by found parcels

Key metrics tracked: API calls, found parcels, pruned points, savings percentage.

### Data Flow

```
KML File → KMLParser → Grid Points → TKGMClient (batch) → Worker → TKGM API
                                                              ↓
                                              Parcel GeoJSON/KML Export
```

## Key Implementation Details

- Grid step converts meters to degrees: `lat_step = meters / 111000`, `lon_step = meters / (111000 * cos(lat))`
- Point-in-polygon: Ray casting algorithm in `KMLParser.point_in_polygon()`
- Parcel deduplication via `ozet` property (e.g., "Bolu-Mudurnu-Merkez-101-5")
- Settings persistence via `QSettings("TKGM", "ParselCekme")`
- 401 errors trigger `auth_error` signal for API key mismatch handling

## Future Development (see TASKS.md)

Planned optimizations in priority order:
- FAZ 1: Batch + Pruning (implemented)
- FAZ 2: Quadtree adaptive sampling
- FAZ 3: Boundary walking (flood-fill from parcel edges)
- FAZ 4: Statistical stopping criteria
- FAZ 5: Computer vision pre-filtering
