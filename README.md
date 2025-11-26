# TKGM Parsel Extractor

TKGM (Tapu ve Kadastro Genel Mudurlugu) API'sinden belirli bir alan icindeki tum parsel verilerini toplu olarak ceken modern desktop uygulamasi.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyQt5](https://img.shields.io/badge/PyQt5-Fluent%20UI-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

[![Deploy to Cloudflare Workers](https://deploy.workers.cloudflare.com/button)](https://deploy.workers.cloudflare.com/?url=https://github.com/mytsx/tkgm-parsel-extractor)

## Ozellikler

- KML dosyasindan alan yukleyebilme
- Cloudflare Workers ile rate limit bypass
- **Akilli tarama**: Bulunan parsellerin icindeki noktalar otomatik elenir (10x-50x daha hizli)
- **Opsiyonel API Key** korumasi
- Modern Fluent Design arayuz
- GeoJSON ve KML formatinda export (tum parsel bilgileri dahil)
- Ilerleme takibi ve log
- Ayarlari otomatik kaydetme

## Kurulum

### 1. Bagimliliklari Yukle

```bash
pip install -r requirements.txt
```

### 2. Cloudflare Worker Deploy Et

1. [Cloudflare Dashboard](https://dash.cloudflare.com) > Workers & Pages > Create Worker
2. `cloudflare-worker.js` icerigini yapistirin
3. "Save and Deploy" tiklayin
4. Worker URL'inizi kopyalayin (ornek: `https://tkgm-proxy.xxx.workers.dev`)

### 3. Uygulamayi Calistir

```bash
python app.py
```

## Kullanim

1. Worker URL'inizi girin
2. API Key girin (opsiyonel - bos birakilabilir)
3. KML dosyanizi secin (taranacak alan)
4. Grid araligini ayarlayin (kucuk = daha hassas, daha yavas)
5. "Taramayi Baslat" butonuna basin
6. Tamamlaninca "Kaydet" ile GeoJSON/KML olarak export edin

## API Key Korumasi (Opsiyonel)

Worker'inizi korumak istiyorsaniz:

1. Cloudflare Dashboard > Worker > Settings > Variables
2. `API_KEY` adinda yeni variable ekleyin (Type: Secret)
3. Degerini belirleyin (ornek: `my-secret-key-123`)
4. Ayni key'i uygulamada "API Key" alanina girin

**Not:** API Key eklemezseniz worker herkese acik olur. Kendi kullanim senaryonuza gore karar verin.

## EXE/APP Build

```bash
pip install pyinstaller
python build.py
```

Cikti: `dist/TKGM-Parsel.app` (macOS) veya `dist/TKGM-Parsel.exe` (Windows)

## Dosya Yapisi

```
├── app.py                 # Ana desktop uygulamasi (PyQt5 + Fluent)
├── cloudflare-worker.js   # Cloudflare Worker proxy kodu
├── tkgm_client.py         # Python API client (CLI kullanim icin)
├── build.py               # PyInstaller build scripti
├── requirements.txt       # Python bagimliliklari
└── KURULUM.md            # Detayli kurulum talimatlari
```

## API Endpointleri (Worker)

```
GET /parsel/{lat}/{lon}    - Tek koordinat sorgulama
POST /batch                - Toplu koordinat sorgulama
```

## Yasal Uyari

Bu arac sadece **yasal amaclar** icin kullanilmalidir:
- Kendi mulklerinizi sorgulama
- Akademik arastirma
- Resmi izinli projeler (madencilik, insaat vb.)

TKGM verilerinin ticari kullanimi icin resmi izin alinmasi gerekmektedir.

## Lisans

MIT License - Detaylar icin [LICENSE](LICENSE) dosyasina bakin.

## Katkida Bulunma

Pull request'ler memnuniyetle karsilanir. Buyuk degisiklikler icin once bir issue acin.
