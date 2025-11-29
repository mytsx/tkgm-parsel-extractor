# TKGM Worker

TKGM (Tapu ve Kadastro Genel Müdürlüğü) API Proxy - Cloudflare Worker

## Özellikler

- Koordinat ile parsel sorgulama
- Toplu (batch) parsel sorgulama
- İl/İlçe/Mahalle listeleri
- Ada/Parsel numarası ile sorgulama
- CORS desteği
- Opsiyonel API Key koruması

## Kurulum

```bash
npm install -g wrangler
wrangler login
wrangler deploy
```

## API Endpoint'leri

### Base URL
```
https://tkgm-parsel-proxy.<account>.workers.dev
```

### İdari Yapı
| Endpoint | Açıklama |
|----------|----------|
| `GET /iller` | Tüm illerin listesi |
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
curl https://tkgm-parsel-proxy.<account>.workers.dev/iller

# Koordinat ile parsel
curl https://tkgm-parsel-proxy.<account>.workers.dev/parsel/40.123/32.456

# Batch sorgu
curl -X POST https://tkgm-parsel-proxy.<account>.workers.dev/batch \
  -H "Content-Type: application/json" \
  -d '{"coordinates": [{"lat": 40.1, "lon": 32.4}, {"lat": 40.2, "lon": 32.5}]}'
```

## Dosya Yapısı

```
├── cloudflare-worker.js  # Ana worker kodu
├── wrangler.toml         # Wrangler yapılandırması
├── workers.json          # Worker metadata
└── workers/              # Ek worker'lar (deprecated)
```

## Lisans

MIT
