# TKGM Parsel Fetching Algorithm

Bu dokuman, TKGM (Tapu Kadastro) API'sinden parsel verisi cekme algoritmasini dil-bagimsiz (language-agnostic) olarak aciklar.

---

## 1. Genel Mimari

```
+------------------+     +-------------------+     +-------------+
|  Uygulama        | --> | Cloudflare Worker | --> | TKGM API    |
|  (Client)        | <-- | (Proxy)           | <-- | (parselsor) |
+------------------+     +-------------------+     +-------------+
        |
        v
  workers.json (config)
```

**Neden Worker Kullaniyoruz?**
- TKGM API direkt erisimde ~100 istekte rate limit (403) uyguluyor
- Worker'lar farkli IP'lerden istek atarak limiti bypass ediyor
- Birden fazla worker ile paralel istek atarak basari oranini artiriyoruz

---

## 2. workers.json Formati

```json
{
  "api_key": "your-secret-api-key",
  "workers": [
    "https://tkgm-proxy-1.example.workers.dev",
    "https://tkgm-proxy-2.example.workers.dev",
    "https://tkgm-proxy-3.example.workers.dev"
  ]
}
```

| Alan | Aciklama |
|------|----------|
| `api_key` | Worker'a erisim icin gizli anahtar (X-API-Key header) |
| `workers` | Worker URL listesi (en az 1 tane) |

---

## 3. Worker API Endpoints

### 3.1. Tekil Sorgu
```
GET /parsel/{lat}/{lon}
Headers: X-API-Key: {api_key}

Response (200): GeoJSON Feature
{
  "type": "Feature",
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[lon, lat], [lon, lat], ...]]
  },
  "properties": {
    "ilAd": "Ankara",
    "ilceAd": "Beypazari",
    "mahalleAd": "Dikmen",
    "adaNo": "101",
    "parselNo": "5",
    "ozet": "Dikmen-101/5",  // Benzersiz ID
    "alan": "1234.56",
    "nitelik": "Tarla",
    ...
  }
}

Response (403): Rate limited
Response (404): Parsel bulunamadi (bos arazi)
Response (401): API key hatali
```

### 3.2. Batch Sorgu (Onerilen)
```
POST /batch
Headers:
  X-API-Key: {api_key}
  Content-Type: application/json

Body:
{
  "coordinates": [
    {"lat": 39.99, "lon": 32.00},
    {"lat": 39.98, "lon": 32.01},
    ...  // Max 20 koordinat
  ]
}

Response (200):
{
  "results": [
    {"status": "found", "data": {...GeoJSON...}},
    {"status": "empty", "data": null},
    {"status": "error", "coord": {"lat": ..., "lon": ...}}
  ],
  "stats": {
    "found": 5,
    "empty": 12,
    "error": 3,
    "total": 20
  }
}
```

---

## 4. Hedged Request Algoritmasi

"Hedged Request" = ayni istegi birden fazla worker'a **paralel** gonder, ilk basarili cevabi kullan.

### 4.1. Paralel Deneme Sayisi Hesaplama

```
worker_count = workers.length

if worker_count < 6:
    parallel_tries = min(3, max(1, (worker_count + 1) / 2))
else:
    parallel_tries = 3
```

| Worker Sayisi | Paralel Deneme |
|---------------|----------------|
| 1             | 1              |
| 2             | 1              |
| 3             | 2              |
| 4             | 2              |
| 5             | 3              |
| 6+            | 3              |

### 4.2. Worker Secimi (Round-Robin + Health Check)

```python
# Pseudocode
def get_healthy_workers(count):
    healthy = []
    checked = 0
    current_time = now()

    while len(healthy) < count and checked < worker_count * 2:
        worker = workers[current_index]
        current_index = (current_index + 1) % worker_count
        checked += 1

        # Cooldown kontrolu
        if worker not in cooldown_map or current_time >= cooldown_map[worker]:
            healthy.append(worker)

    # Yeterli saglikli worker yoksa, cooldown'dakileri de ekle
    if len(healthy) < count:
        for worker in workers:
            if worker not in healthy:
                healthy.append(worker)
            if len(healthy) >= count:
                break

    return healthy[:count]
```

### 4.3. Hedged Request Akisi

```
function hedged_request(url_path, method, payload):
    workers = get_healthy_workers(parallel_tries)

    # Paralel istek at
    futures = []
    for worker in workers:
        futures.append(async_request(worker + url_path, method, payload))

    # Ilk basarili cevabi bekle
    for future in as_completed(futures):
        response, status_code = future.result()
        worker = future.worker

        if status_code == 200:
            mark_worker_success(worker)
            cancel_remaining_futures()
            return response

        if status_code == 403:  # Rate limited
            mark_worker_rate_limited(worker)  # 5 dk cooldown
        else:
            mark_worker_fail(worker)

    # Hicbiri basarili olmadi
    return null
```

---

## 5. Worker Health Tracking

### 5.1. Veri Yapilari

```
cooldown_map = {}      // {worker_url: cooldown_end_timestamp}
success_count = {}     // {worker_url: int}
fail_count = {}        // {worker_url: int}

COOLDOWN_SECONDS = 300  // 5 dakika
```

### 5.2. Fonksiyonlar

```python
def mark_worker_success(worker):
    success_count[worker] += 1
    if worker in cooldown_map:
        del cooldown_map[worker]  # Cooldown'dan cikar

def mark_worker_rate_limited(worker):
    fail_count[worker] += 1
    cooldown_map[worker] = now() + COOLDOWN_SECONDS

def mark_worker_fail(worker):
    fail_count[worker] += 1
    # Cooldown'a almiyoruz, sadece fail sayiyoruz
```

---

## 6. Batch + Pruning Tarama Algoritmasi

Bu kisim alansal tarama icin (grid uzerinden butun parselleri bulmak).

### 6.1. Grid Noktalari Olusturma

```python
def generate_grid_points(polygon, step_meters):
    # Metre -> derece donusumu
    lat_step = step_meters / 111000
    lon_step = step_meters / (111000 * cos(center_lat))

    points = []
    for lat in range(min_lat, max_lat, lat_step):
        for lon in range(min_lon, max_lon, lon_step):
            if point_in_polygon(lat, lon, polygon):
                points.append((lat, lon))

    return points
```

### 6.2. Tarama Akisi

```
1. Grid noktalari olustur
2. Noktalardan batch al (15-20 koordinat)
3. Batch sorgusu at
4. Bulunan her parsel icin:
   a. Parsel geometrisini al
   b. Kalan noktalari filtrele:
      - Parsel icinde kalan noktalari cikar (PRUNING)
5. Kalan nokta varsa 2'ye don
6. Bitir, sonuclari kaydet
```

### 6.3. Pruning Optimizasyonu

```python
def prune_points(remaining_points, found_parcel_geometry):
    pruned = []
    for point in remaining_points:
        if not point_in_polygon(point, found_parcel_geometry):
            pruned.append(point)  # Parsel disinda, taranmaya devam
    return pruned
```

**Neden Pruning?**
- Bir parsel bulunca, icindeki diger grid noktalarini sorgulamaya gerek yok
- %60-80 API tasarrufu saglar

---

## 7. HTTP Header'lar

```
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
Accept: application/json
X-API-Key: {api_key}  // workers.json'dan
```

---

## 8. Timeout Degerleri

| Istek Tipi | Timeout |
|------------|---------|
| Tekil GET  | 15 saniye |
| Batch POST | 60 saniye |

---

## 9. Hata Kodlari

| Kod | Anlami | Aksiyon |
|-----|--------|---------|
| 200 | Basarili | Veriyi isle |
| 401 | API key hatali | Kullaniciya bildir |
| 403 | Rate limited | Worker'i 5dk cooldown'a al |
| 404 | Parsel yok | Bos arazi, skip |
| 429 | Too many requests | Batch'te rate limit |
| 5xx | Server hatasi | Retry veya skip |

---

## 10. Ornek Implementasyon (Pseudocode)

```python
class TKGMClient:
    def __init__(self, config_path):
        config = json.load(config_path)
        self.workers = config["workers"]
        self.api_key = config["api_key"]
        self.cooldown_map = {}
        self.current_index = 0

        # Paralel deneme sayisi
        n = len(self.workers)
        self.parallel_tries = min(3, max(1, (n + 1) // 2)) if n < 6 else 3

    def get_parcel(self, lat, lon):
        response = self._hedged_request(f"/parsel/{lat}/{lon}")
        if response and response.status == 200:
            return response.json()
        return None

    def get_batch(self, coordinates):
        """coordinates: [(lat, lon), ...]"""
        payload = {
            "coordinates": [{"lat": c[0], "lon": c[1]} for c in coordinates]
        }
        response = self._hedged_request("/batch", "POST", payload)
        if response and response.status == 200:
            return response.json()
        return None

    def _hedged_request(self, path, method="GET", payload=None):
        workers = self._get_healthy_workers(self.parallel_tries)

        # Paralel istek (dile gore degisir: async/await, threads, goroutines, etc.)
        futures = parallel_map(
            lambda w: http_request(w + path, method, payload, self.api_key),
            workers
        )

        for result in as_completed(futures):
            if result.status == 200:
                return result
            elif result.status == 403:
                self._cooldown(result.worker)

        return None
```

---

## 11. Diger Dillere Port Notlari

### Go
- `sync.WaitGroup` veya channel'lar ile paralel istek
- `time.AfterFunc` ile cooldown timer

### JavaScript/Node
- `Promise.race()` veya `Promise.any()` ile ilk basariyi al
- `setTimeout` ile cooldown

### Rust
- `tokio::select!` macro ile race
- `std::collections::HashMap` ile cooldown tracking

### Java
- `CompletableFuture.anyOf()` ile race
- `ConcurrentHashMap` ile thread-safe cooldown

---

## 12. Onemli Notlar

1. **Benzersiz ID**: Parsel `properties.ozet` alani benzersiz, ornegin "Dikmen-101/5"
2. **Koordinat Sistemi**: WGS84 (EPSG:4326), lat/lon sirasi
3. **Batch Limiti**: Max 20 koordinat per batch (worker tarafinda)
4. **Gunluk Limit**: TKGM ~1000-2000 sorgu/gun limit koyuyor (worker basina)
5. **Geometry Format**: GeoJSON Polygon, coordinates = [[[lon, lat], [lon, lat], ...]]
