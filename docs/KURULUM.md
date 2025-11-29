# TKGM Veri Cekme Araci - Kurulum

## 1. Cloudflare Workers Kurulumu

### Adim 1: Cloudflare Hesabi
1. https://dash.cloudflare.com adresine gidin
2. Hesabiniz yoksa ucretsiz hesap olusturun
3. Workers & Pages > Create Application > Create Worker

### Adim 2: Worker Deploy
1. Worker adinizi girin (ornek: `tkgm-proxy`)
2. "Deploy" butonuna basin
3. "Edit code" butonuna basin
4. `cloudflare-worker.js` icerigini yapistirin
5. "Save and Deploy" basin

### Adim 3: URL'inizi Alin
Deploy sonrasi size bir URL verilecek:
```
https://tkgm-proxy.YOUR_SUBDOMAIN.workers.dev
```

## 2. Python Client Kurulumu

```bash
# Gerekli kutuphane
pip install requests

# Script'i calistir
python tkgm_client.py
```

## 3. Kullanim

### tkgm_client.py Duzenleme

```python
# Worker URL'inizi guncelleyin (satir 20 ve 242)
WORKER_URL = "https://tkgm-proxy.YOUR_SUBDOMAIN.workers.dev"
```

### Ornek: Tek Nokta Sorgulama

```python
from tkgm_client import TKGMClient

client = TKGMClient(worker_url="https://tkgm-proxy.xxx.workers.dev")
result = client.get_parsel(40.150603, 31.941933)
print(result)
```

### Ornek: Alan Tarama (Maden Ruhsat Alani)

```python
from tkgm_client import TKGMClient, BoundingBox

client = TKGMClient(worker_url="https://tkgm-proxy.xxx.workers.dev")

# Ruhsat alaninizin sinirlarini girin
bbox = BoundingBox(
    min_lat=40.1500,  # Guney siniri
    max_lat=40.1600,  # Kuzey siniri
    min_lon=31.9400,  # Bati siniri
    max_lon=31.9500   # Dogu siniri
)

# Alani tara (30m aralikla)
parcels = client.scan_area(
    bbox=bbox,
    step_meters=30,
    batch_size=10,
    delay_between_batches=0.5
)

# Kaydet
client.save_geojson(parcels, "ruhsat_alani.geojson")
client.save_kml(parcels, "ruhsat_alani.kml")
```

## 4. Parametreler

| Parametre | Aciklama | Onerilen Deger |
|-----------|----------|----------------|
| step_meters | Grid araligi | 20-50m (kucuk = daha fazla sorgu) |
| batch_size | Toplu sorgu sayisi | 10-20 |
| delay_between_batches | Bekleme suresi | 0.5-1 saniye |

## 5. Cloudflare Workers Limitleri

Ucretsiz plan:
- 100,000 istek/gun
- 10ms CPU time/istek

Bu limitler cogu maden projesi icin yeterli olmali.

## 6. Cikti Dosyalari

- `parcels.geojson` - GIS yazilimlarinda kullanilabilir (QGIS, ArcGIS)
- `parcels.kml` - Google Earth'te goruntulenebilir

## Sorun Giderme

**Worker calismiyorsa:**
- Cloudflare Dashboard > Workers > Logs'u kontrol edin

**Rate limit hatasi aliyorsaniz:**
- `delay_between_batches` degerini artirin
- `batch_size` degerini azaltin
