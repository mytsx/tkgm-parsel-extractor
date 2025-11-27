#!/usr/bin/env python3
"""
TKGM Parsel Veri Çekme Aracı
Cloudflare Workers proxy üzerinden çalışır

Kullanım:
1. Cloudflare Worker'ı deploy edin
2. WORKER_URL'i güncelleyin
3. Script'i çalıştırın
"""

import json
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

# Cloudflare Worker URL'inizi buraya yazın
WORKER_URL = "https://your-worker.your-subdomain.workers.dev"

@dataclass
class BoundingBox:
    """Coğrafi sınır kutusu"""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

class TKGMClient:
    """TKGM API istemcisi - Worker proxy veya direkt API uzerinden parsel sorgular."""

    def __init__(self, worker_url: str = None, use_direct: bool = False):
        """
        Args:
            worker_url: Cloudflare Worker URL'i
            use_direct: True ise direkt TKGM API'sine gider (rate limit var)
        """
        self.worker_url = worker_url or WORKER_URL
        self.use_direct = use_direct
        self.direct_url = "https://cbsapi.tkgm.gov.tr/megsiswebapi.v3.1/api/parsel"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        })

    def get_parsel(self, lat: float, lon: float) -> Optional[Dict]:
        """Tek bir koordinat için parsel bilgisi al"""
        try:
            if self.use_direct:
                url = f"{self.direct_url}/{lat}/{lon}/"
                self.session.headers.update({
                    "Referer": "https://parselsorgu.tkgm.gov.tr/",
                    "Origin": "https://parselsorgu.tkgm.gov.tr"
                })
            else:
                url = f"{self.worker_url}/parsel/{lat}/{lon}"

            response = self.session.get(url, timeout=30)

            if response.status_code == 200:
                return response.json()
            print(f"Hata: {response.status_code} - {lat}, {lon}")
            return None
        except requests.RequestException as e:
            print(f"İstek hatası: {e}")
            return None

    def get_batch(self, coordinates: List[Dict[str, float]]) -> List[Dict]:
        """Toplu koordinat sorgulama (sadece Worker ile)"""
        if self.use_direct:
            # Direkt modda tek tek sorgula
            results = []
            for coord in coordinates:
                result = self.get_parsel(coord['lat'], coord['lon'])
                if result:
                    results.append(result)
                time.sleep(0.5)  # Rate limit için bekle
            return results

        try:
            response = self.session.post(
                f"{self.worker_url}/batch",
                json={"coordinates": coordinates},
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])
            print(f"Batch hata: {response.status_code}")
            return []
        except requests.RequestException as e:
            print(f"Batch istek hatası: {e}")
            return []

    def generate_grid(self, bbox: BoundingBox, step_meters: float = 50) -> List[Tuple[float, float]]:
        """
        Verilen sınır kutusu içinde grid noktaları oluştur

        Args:
            bbox: Sınır kutusu
            step_meters: Grid adım mesafesi (metre)

        Returns:
            [(lat, lon), ...] koordinat listesi
        """
        # Metre -> derece dönüşümü (yaklaşık)
        # 1 derece lat ≈ 111km
        # 1 derece lon ≈ 111km * cos(lat)

        center_lat = (bbox.min_lat + bbox.max_lat) / 2
        lat_step = step_meters / 111000
        lon_step = step_meters / (111000 * math.cos(math.radians(center_lat)))

        points = []
        lat = bbox.min_lat
        while lat <= bbox.max_lat:
            lon = bbox.min_lon
            while lon <= bbox.max_lon:
                points.append((lat, lon))
                lon += lon_step
            lat += lat_step

        return points

    def scan_area(
        self,
        bbox: BoundingBox,
        step_meters: float = 50,
        batch_size: int = 10,
        delay_between_batches: float = 1.0
    ) -> Dict[str, Dict]:
        """
        Belirli bir alanı tara ve tüm parselleri bul

        Args:
            bbox: Taranacak alan
            step_meters: Grid adım mesafesi
            batch_size: Her batch'teki sorgu sayısı
            delay_between_batches: Batch'ler arası bekleme (saniye)

        Returns:
            {parsel_id: parsel_data} dictionary
        """
        points = self.generate_grid(bbox, step_meters)
        print(f"Toplam {len(points)} nokta taranacak")

        all_parcels = {}

        # Noktaları batch'lere böl
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i + batch_size]
            coordinates = [{"lat": p[0], "lon": p[1]} for p in batch_points]

            print(f"Batch {i//batch_size + 1}/{math.ceil(len(points)/batch_size)} işleniyor...")

            if self.use_direct:
                # Direkt modda tek tek
                for coord in coordinates:
                    result = self.get_parsel(coord['lat'], coord['lon'])
                    if result and 'properties' in result:
                        parsel_id = result['properties'].get('ozet', f"{coord['lat']}_{coord['lon']}")
                        if parsel_id not in all_parcels:
                            all_parcels[parsel_id] = result
                            print(f"  Yeni parsel bulundu: {parsel_id}")
                    time.sleep(0.3)
            else:
                # Worker ile batch
                results = self.get_batch(coordinates)
                for idx, result in enumerate(results):
                    if result and 'properties' in result:
                        if idx < len(coordinates):
                            coord = coordinates[idx]
                            fallback = f"{coord.get('lat', 'unknown')}_{coord.get('lon', 'unknown')}"
                        else:
                            fallback = f"unknown_index_{idx}"
                        parsel_id = result['properties'].get('ozet') or fallback
                        if parsel_id not in all_parcels:
                            all_parcels[parsel_id] = result
                            print(f"  Yeni parsel bulundu: {parsel_id}")

            time.sleep(delay_between_batches)

        print(f"\nToplam {len(all_parcels)} benzersiz parsel bulundu")
        return all_parcels

    def parcels_to_geojson(self, parcels: Dict[str, Dict]) -> Dict:
        """Parselleri GeoJSON FeatureCollection'a dönüştür"""
        features = list(parcels.values())
        return {
            "type": "FeatureCollection",
            "features": features
        }

    def save_geojson(self, parcels: Dict[str, Dict], filename: str):
        """Parselleri GeoJSON dosyasına kaydet"""
        geojson = self.parcels_to_geojson(parcels)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)
        print(f"GeoJSON kaydedildi: {filename}")

    def save_kml(self, parcels: Dict[str, Dict], filename: str):
        """Parselleri KML dosyasına kaydet"""
        kml_header = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>TKGM Parseller</name>
<Style id="parselStyle">
    <LineStyle><color>ff0000ff</color><width>2</width></LineStyle>
    <PolyStyle><color>4d0000ff</color></PolyStyle>
</Style>
'''
        kml_footer = '''</Document>
</kml>'''

        placemarks = []
        for parsel_id, parsel in parcels.items():
            props = parsel.get('properties', {})
            geom = parsel.get('geometry', {})

            if geom.get('type') != 'Polygon':
                continue

            coords = geom.get('coordinates', [[]])[0]
            coord_str = ' '.join([f"{c[0]},{c[1]},0" for c in coords])

            name = props.get('ozet', parsel_id)
            description = f"""
            İl: {props.get('ilAd', '')}
            İlçe: {props.get('ilceAd', '')}
            Mahalle: {props.get('mahalleAd', '')}
            Ada: {props.get('adaNo', '')}
            Parsel: {props.get('parselNo', '')}
            Alan: {props.get('alan', '')}
            Nitelik: {props.get('nitelik', '')}
            """

            placemark = f'''<Placemark>
    <name>{name}</name>
    <description><![CDATA[{description}]]></description>
    <styleUrl>#parselStyle</styleUrl>
    <Polygon>
        <outerBoundaryIs>
            <LinearRing>
                <coordinates>{coord_str}</coordinates>
            </LinearRing>
        </outerBoundaryIs>
    </Polygon>
</Placemark>
'''
            placemarks.append(placemark)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(kml_header)
            f.write('\n'.join(placemarks))
            f.write(kml_footer)

        print(f"KML kaydedildi: {filename}")


def main():
    """Örnek kullanım"""

    # Worker URL'inizi buraya yazın
    client = TKGMClient(
        worker_url="https://your-worker.your-subdomain.workers.dev",
        use_direct=False  # True yaparsanız direkt TKGM'ye gider (rate limit!)
    )

    # Örnek: Tek nokta sorgulama
    print("=== Tek Nokta Sorgusu ===")
    result = client.get_parsel(40.150603498577404, 31.941933631896973)
    if result:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    # Örnek: Belirli bir alanı tarama
    print("\n=== Alan Taraması ===")
    bbox = BoundingBox(
        min_lat=40.1500,
        max_lat=40.1520,
        min_lon=31.9400,
        max_lon=31.9450
    )

    parcels = client.scan_area(
        bbox=bbox,
        step_meters=30,      # 30 metre aralıklarla tara
        batch_size=10,       # Her seferde 10 nokta
        delay_between_batches=0.5  # Batch'ler arası 0.5 saniye
    )

    # Sonuçları kaydet
    if parcels:
        client.save_geojson(parcels, "parcels.geojson")
        client.save_kml(parcels, "parcels.kml")


if __name__ == "__main__":
    main()
