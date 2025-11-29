# TKGM Worker

TKGM (Tapu ve Kadastro Genel Müdürlüğü) API Proxy - Cloudflare Workers

![Cloudflare Workers](https://img.shields.io/badge/Cloudflare-Workers-orange)
![Version](https://img.shields.io/badge/Version-2.0-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

[![Deploy to Cloudflare Workers](https://deploy.workers.cloudflare.com/button)](https://deploy.workers.cloudflare.com/?url=https://github.com/mytsx/tkgm-parsel-extractor)

## Özellikler

- Koordinat ile parsel sorgulama
- Toplu (batch) parsel sorgulama (20 koordinat/istek)
- İl/İlçe/Mahalle listeleri (GeoJSON formatında)
- Ada/Parsel numarası ile sorgulama
- CORS desteği
- API Key koruması

## Workers

12 adet load-balanced worker:

```
https://tkgm-proxy-1.example.workers.dev
https://tkgm-proxy-2.example.workers.dev
...
https://tkgm-proxy-12.example.workers.dev
```

## Kurulum

```bash
npm install -g wrangler
wrangler login

# Tek worker deploy
wrangler deploy

# Tüm worker'lara deploy
seq 1 12 | xargs -I {} wrangler deploy --name "tkgm-proxy-{}"

# API Key ekleme
wrangler secret put API_KEY --name "tkgm-proxy-1"
```

## API Endpoint'leri

### Authentication
Tüm isteklerde `X-API-Key` header'ı gerekli:
```bash
curl -H "X-API-Key: YOUR_API_KEY" https://tkgm-proxy-1.example.workers.dev/iller
```

### İdari Yapı
| Endpoint | Açıklama |
|----------|----------|
| `GET /iller` | Tüm illerin listesi (81 il) |
| `GET /ilceler/{ilId}` | İlin ilçeleri |
| `GET /mahalleler/{ilceId}` | İlçenin mahalleleri |
| `GET /parsel-by-ada/{mahalleId}/{ada}/{parsel}` | Ada/Parsel sorgusu |

### Parsel Sorgulama
| Endpoint | Açıklama |
|----------|----------|
| `GET /parsel/{lat}/{lon}` | Koordinat ile parsel |
| `POST /batch` | Toplu koordinat sorgusu |

## Örnek Kullanım

```bash
# İl listesi
curl -H "X-API-Key: $API_KEY" \
  https://tkgm-proxy-1.example.workers.dev/iller

# Bolu'nun ilçeleri (il ID: 36)
curl -H "X-API-Key: $API_KEY" \
  https://tkgm-proxy-1.example.workers.dev/ilceler/36

# Koordinat ile parsel
curl -H "X-API-Key: $API_KEY" \
  https://tkgm-proxy-1.example.workers.dev/parsel/40.123/32.456

# Batch sorgu
curl -X POST \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"coordinates": [{"lat": 40.1, "lon": 32.4}, {"lat": 40.2, "lon": 32.5}]}' \
  https://tkgm-proxy-1.example.workers.dev/batch
```

## Dosya Yapısı

```
├── cloudflare-worker.js  # Ana worker kodu
├── wrangler.toml         # Wrangler yapılandırması
├── workers.json          # Worker metadata
├── docs/                 # Dokümantasyon
│   ├── ALGORITHM.md
│   ├── KURULUM.md
│   └── TASKS.md
└── backup/               # Eski Python uygulama kodu
```

## Lisans

MIT
