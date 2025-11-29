# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TKGM Worker - Cloudflare Worker that proxies requests to the Turkish Land Registry (TKGM) API. Provides parcel queries by coordinates, batch queries, and administrative data (provinces, districts, neighborhoods).

## Commands

```bash
# Deploy worker
wrangler deploy

# Test locally
wrangler dev

# Add API key (optional)
wrangler secret put API_KEY

# Delete worker
wrangler delete
```

## Architecture

### cloudflare-worker.js

Single worker file with all endpoints:

**İdari Yapı (Administrative)**
- `GET /iller` - All provinces list
- `GET /ilceler/{ilId}` - Districts of a province
- `GET /mahalleler/{ilceId}` - Neighborhoods of a district
- `GET /parsel-by-ada/{mahalleId}/{ada}/{parsel}` - Parcel by block/lot number

**Parsel Sorgulama (Parcel Query)**
- `GET /parsel/{lat}/{lon}` - Single coordinate query
- `POST /batch` - Batch query (up to 20 coordinates)

### Key Implementation Details

- Uses two TKGM API versions: v3 (administrative) and v3.1 (parcel)
- Staggered parallel requests in batch endpoint to avoid rate limits
- Optional API Key protection via `X-API-Key` header
- CORS enabled for all origins

### Configuration

- `wrangler.toml` - Worker name, compatibility date
- `workers.json` - Metadata, endpoints list

## Files

```
├── cloudflare-worker.js  # Main worker code
├── wrangler.toml         # Wrangler config
├── workers.json          # Worker metadata
├── workers/              # Additional workers (deprecated)
└── backup/               # Old app code (Python desktop app)
```
