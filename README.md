# TKGM Parsel Extractor

TKGM (Tapu ve Kadastro Genel Mudurlugu) API'sinden belirli bir alan icindeki tum parsel verilerini toplu olarak ceken modern desktop uygulamasi.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyQt5](https://img.shields.io/badge/PyQt5-Fluent%20UI-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Ozellikler

- KML dosyasindan alan yukleyebilme
- Cloudflare Workers ile rate limit bypass
- Modern Fluent Design arayuz
- GeoJSON ve KML formatinda export
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
2. KML dosyanizi secin (taranacak alan)
3. Grid araligini ayarlayin (kucuk = daha hassas, daha yavas)
4. "Taramayi Baslat" butonuna basin
5. Tamamlaninca "Kaydet" ile GeoJSON/KML olarak export edin

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
