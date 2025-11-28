#!/usr/bin/env python3
"""
TKGM Parsel Veri Cekme - Modern Desktop Uygulamasi
PyQt5 + Fluent Design
"""

import sys
import json
import logging
import math
import time
from collections import deque
from pathlib import Path
from xml.etree import ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings

# Fluent Widgets
from qfluentwidgets import (
    PushButton, PrimaryPushButton, LineEdit, BodyLabel, TitleLabel,
    SubtitleLabel, CardWidget, ProgressBar, TextEdit,
    SpinBox, DoubleSpinBox, InfoBar, InfoBarPosition, FluentIcon,
    setTheme, Theme
)

import requests


# ==================== VERI MODELLERI ====================

@dataclass
class BoundingBox:
    """Cografi sinir kutusu (bounding box)."""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float


# ==================== KML PARSER ====================

class KMLParser:
    """KML dosyalarini parse eden yardimci sinif."""

    @staticmethod
    def parse(kml_path: str) -> List[List[Tuple[float, float]]]:
        """KML dosyasindan polygon koordinatlarini cikarir."""
        tree = ET.parse(kml_path)
        root = tree.getroot()
        polygons = []

        for coordinates in root.iter():
            if coordinates.tag.endswith('coordinates'):
                coords_text = coordinates.text.strip()
                polygon = KMLParser._parse_coordinates(coords_text)
                if polygon:
                    polygons.append(polygon)
        return polygons

    @staticmethod
    def _parse_coordinates(coords_text: str) -> List[Tuple[float, float]]:
        """Koordinat stringini (lat, lon) tuple listesine donusturur."""
        coords = []
        for coord in coords_text.split():
            parts = coord.strip().split(',')
            if len(parts) >= 2:
                lon = float(parts[0])
                lat = float(parts[1])
                coords.append((lat, lon))
        return coords

    @staticmethod
    def get_bounding_box(polygon: List[Tuple[float, float]]) -> BoundingBox:
        """Polygon icin bounding box hesaplar."""
        lats = [p[0] for p in polygon]
        lons = [p[1] for p in polygon]
        return BoundingBox(min(lats), max(lats), min(lons), max(lons))

    @staticmethod
    def point_in_polygon(lat: float, lon: float, polygon: List[Tuple[float, float]]) -> bool:
        """Ray casting algoritmasi ile noktanin polygon icinde olup olmadigini kontrol eder."""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            if ((polygon[i][1] > lon) != (polygon[j][1] > lon)) and \
               (lat < (polygon[j][0] - polygon[i][0]) * (lon - polygon[i][1]) /
                (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
                inside = not inside
            j = i
        return inside


# ==================== TKGM CLIENT ====================

class TKGMClient:
    """Cloudflare Worker uzerinden TKGM API istemcisi."""

    REQUEST_TIMEOUT = 30       # Tekil istek timeout (saniye)
    BATCH_REQUEST_TIMEOUT = 120  # Batch istek timeout (saniye)

    def __init__(self, worker_url: str, api_key: str = None):
        """Worker URL ve opsiyonel API key ile istemci olusturur."""
        self.worker_url = worker_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        })
        if api_key:
            self.session.headers["X-API-Key"] = api_key

    def get_parsel(self, lat: float, lon: float) -> tuple:
        """Returns (data, error_code) tuple"""
        try:
            url = f"{self.worker_url}/parsel/{lat}/{lon}"
            response = self.session.get(url, timeout=self.REQUEST_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                if 'properties' in data:
                    return data, None
            elif response.status_code == 401:
                return None, 401
            return None, response.status_code
        except requests.RequestException as e:
            return None, str(e)

    def get_batch(self, coordinates: list) -> tuple:
        """
        Batch sorgu - birden fazla koordinati tek istekte sorgula
        Args:
            coordinates: [(lat, lon), (lat, lon), ...] listesi
        Returns:
            (results_list, error_code) tuple
            results_list: [{data}, {data}, ...] veya None'lar
        """
        try:
            url = f"{self.worker_url}/batch"
            payload = {
                "coordinates": [{"lat": c[0], "lon": c[1]} for c in coordinates]
            }
            response = self.session.post(url, json=payload, timeout=self.BATCH_REQUEST_TIMEOUT)

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                # Her sonucu kontrol et, gecerli parsel mi?
                processed = []
                for r in results:
                    if r and isinstance(r, dict) and 'properties' in r:
                        processed.append(r)
                    else:
                        processed.append(None)
                return processed, None
            if response.status_code == 401:
                return None, 401
            return None, response.status_code
        except requests.RequestException as e:
            return None, str(e)


# ==================== QUADTREE ====================

class QuadTreeNode:
    """Quadtree dugumu - alani adaptif bolmek icin kullanilir."""

    # Minimum hucre boyutu (metre) - bunun altina bolunmez
    MIN_CELL_SIZE = 20

    def __init__(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float, depth: int = 0):
        """Quadtree dugumu olusturur."""
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.depth = depth
        self.children = []  # 4 alt dugum (NW, NE, SW, SE)
        self.is_leaf = True
        self.is_empty = None  # None=bilinmiyor, True=bos, False=parsel var
        self.parcel_ids = set()  # Bu hucrede bulunan parsel ID'leri

    def get_center(self) -> Tuple[float, float]:
        """Hucrenin merkez koordinatini dondurur."""
        return ((self.min_lat + self.max_lat) / 2, (self.min_lon + self.max_lon) / 2)

    def get_corners(self) -> List[Tuple[float, float]]:
        """Hucrenin 4 kosesini dondurur."""
        return [
            (self.min_lat, self.min_lon),  # SW
            (self.min_lat, self.max_lon),  # SE
            (self.max_lat, self.min_lon),  # NW
            (self.max_lat, self.max_lon),  # NE
        ]

    def get_sample_points(self) -> List[Tuple[float, float]]:
        """Hucreyi test etmek icin 5 nokta dondurur (4 kose + merkez)."""
        center = self.get_center()
        return self.get_corners() + [center]

    def get_cell_size_meters(self) -> float:
        """Hucrenin yaklasik boyutunu metre cinsinden dondurur."""
        center_lat = (self.min_lat + self.max_lat) / 2
        lat_size = (self.max_lat - self.min_lat) * 111000
        lon_size = (self.max_lon - self.min_lon) * 111000 * math.cos(math.radians(center_lat))
        return max(lat_size, lon_size)

    def can_subdivide(self) -> bool:
        """Hucrenin daha fazla bolunup bolunemeyecegini kontrol eder."""
        return self.get_cell_size_meters() > self.MIN_CELL_SIZE * 2

    def subdivide(self):
        """Hucreyi 4 alt hucreye boler."""
        if not self.can_subdivide():
            return

        mid_lat = (self.min_lat + self.max_lat) / 2
        mid_lon = (self.min_lon + self.max_lon) / 2

        self.children = [
            QuadTreeNode(self.min_lat, mid_lat, self.min_lon, mid_lon, self.depth + 1),  # SW
            QuadTreeNode(self.min_lat, mid_lat, mid_lon, self.max_lon, self.depth + 1),  # SE
            QuadTreeNode(mid_lat, self.max_lat, self.min_lon, mid_lon, self.depth + 1),  # NW
            QuadTreeNode(mid_lat, self.max_lat, mid_lon, self.max_lon, self.depth + 1),  # NE
        ]
        self.is_leaf = False


# ==================== WORKER THREAD ====================

class ScanWorker(QThread):
    """Arka planda alan taramasi yapan worker thread - Quadtree + Grid + Boundary Walking hibrit."""

    # Worker tarafinda MAX_BATCH_COORDS=20 limiti var, bu degerden kucuk tutulmali
    BATCH_SIZE = 15

    # Boundary walking parametreleri
    EDGE_STEP_METERS = 10  # Kenar boyunca adim (metre)
    OUTWARD_STEP_METERS = 8  # Kenardan disa adim (metre)
    BOUNDARY_WALK_MAX_ITERATIONS = 5  # Sonsuz dongu korunmasi
    MIN_EDGE_LENGTH_METERS = 1  # Bu uzunluktan kisa kenarlar atlanir

    # Gap filling parametreleri
    GAP_FILL_STEP_DIVISOR = 2  # step_meters / bu deger = gap fill adimi
    MIN_GAP_FILL_STEP = 5  # Minimum gap fill grid adimi (metre)

    # Refine (detaylandirma) parametreleri
    MIN_REFINE_STEP = 5  # Minimum detaylandirma grid adimi (metre)
    REFINE_STEP_DIVISOR = 3  # step_meters / bu deger = detaylandirma adimi

    progress = pyqtSignal(int, int)  # current, total
    stats_update = pyqtSignal(dict)  # Detayli istatistikler
    log = pyqtSignal(str)
    parcel_found = pyqtSignal(str, dict)
    finished_scan = pyqtSignal(int)
    auth_error = pyqtSignal()  # 401 hatasi icin

    def __init__(self, client: TKGMClient, polygons: List, step_meters: int, delay: float):
        """Tarama worker'i olusturur."""
        super().__init__()
        self.client = client
        self.polygons = polygons
        self.step_meters = step_meters
        self.delay = delay
        self.is_running = True

        # Tarama istatistikleri
        self.found_ids = set()
        self.found_parcels = []  # Bulunan parsel geometrileri (pruning icin)
        self.api_calls = 0
        self.cells_processed = 0
        self.cells_skipped = 0
        self.boundary_walks = 0
        self.gap_fill_points = 0
        self.total_points_queried = 0
        self.points_remaining = 0
        self.current_phase = ""
        self.null_responses = 0  # API'den parsel bulunamadi yaniti
        self.failed_points = 0   # Ag hatasi nedeniyle sorgulanamayan noktalar

    @staticmethod
    def get_parcel_id(result: dict, point: Tuple[float, float]) -> str:
        """Parsel ID'sini dondurur (ozet veya koordinat bazli fallback)."""
        if result and 'properties' in result:
            ozet = result['properties'].get('ozet')
            if ozet:
                return ozet
        return f"{point[0]:.6f}_{point[1]:.6f}"

    @staticmethod
    def geojson_coords_to_poly(coords: list) -> List[Tuple[float, float]]:
        """GeoJSON koordinatlarini (lon, lat) -> (lat, lon) tuple listesine donusturur."""
        return [(c[1], c[0]) for c in coords]

    def _process_found_parcel(self, result: dict, point: Tuple[float, float]) -> Tuple[str, bool]:
        """
        Parsel sonucunu isler. Bulunduysa ID'sini ve yeni olup olmadigini dondurur.
        Returns: (parsel_id, is_new)
        """
        parsel_id = self.get_parcel_id(result, point)
        is_new = False

        if parsel_id not in self.found_ids:
            is_new = True
            self.found_ids.add(parsel_id)
            self.parcel_found.emit(parsel_id, result)
            props = result['properties']
            self.log.emit(f"  ✓ {parsel_id} - {props.get('nitelik', '')} ({props.get('alan', '')})")

            # Geometriyi kaydet (pruning icin) - en az 3 nokta olmali
            if 'geometry' in result and result['geometry'].get('type') == 'Polygon':
                coord_list = result['geometry'].get('coordinates') or []
                if coord_list and len(coord_list[0]) >= 3:
                    self.found_parcels.append(self.geojson_coords_to_poly(coord_list[0]))

        return parsel_id, is_new

    def _emit_stats(self):
        """Guncel istatistikleri UI'a gonderir."""
        self.stats_update.emit({
            "phase": self.current_phase,
            "found": len(self.found_ids),
            "api_calls": self.api_calls,
            "queried": self.total_points_queried,
            "remaining": self.points_remaining,
            "null_responses": self.null_responses,
            "failed_points": self.failed_points
        })

    def run(self):
        """Hibrit Quadtree + Grid + Boundary Walking + Gap Fill tarama algoritmasi."""
        self.log.emit("=== Hibrit Tarama Algoritmasi ===")
        self.log.emit(f"Grid boyutu: {self.step_meters}m, Minimum hucre: {QuadTreeNode.MIN_CELL_SIZE}m")

        found_count = 0

        for poly_idx, polygon in enumerate(self.polygons):
            if not self.is_running:
                break

            self.log.emit(f"\n{'='*50}")
            self.log.emit(f"Polygon {poly_idx + 1}/{len(self.polygons)}")
            self.log.emit("="*50)

            # === FAZ 1: Quadtree + Grid Tarama ===
            self.log.emit("\n[FAZ 1] Quadtree + Grid tarama basliyor...")
            phase1_count = self._phase1_quadtree_scan(polygon)
            found_count += phase1_count
            self.log.emit(f"[FAZ 1] Tamamlandi: {phase1_count} parsel bulundu")

            if not self.is_running:
                break

            # === FAZ 2: Boundary Walking ===
            self.log.emit("\n[FAZ 2] Sinir takibi (Boundary Walking) basliyor...")
            phase2_count = self._phase2_boundary_walk(polygon)
            found_count += phase2_count
            self.log.emit(f"[FAZ 2] Tamamlandi: {phase2_count} yeni parsel bulundu")

            if not self.is_running:
                break

            # === FAZ 3: Gap Filling ===
            self.log.emit("\n[FAZ 3] Bosluk doldurma (Gap Fill) basliyor...")
            phase3_count = self._phase3_gap_fill(polygon)
            found_count += phase3_count
            self.log.emit(f"[FAZ 3] Tamamlandi: {phase3_count} yeni parsel bulundu")

        if not self.is_running:
            self.log.emit("Tarama durduruldu")

        self.progress.emit(100, 100)
        self._log_summary(found_count)
        self.finished_scan.emit(found_count)

    def _phase1_quadtree_scan(self, polygon: List[Tuple[float, float]]) -> int:
        """FAZ 1: Quadtree + Grid hibrit tarama."""
        self.current_phase = "Quadtree"
        found_count = 0

        # Polygon icin quadtree olustur
        bbox = KMLParser.get_bounding_box(polygon)
        root = QuadTreeNode(bbox.min_lat, bbox.max_lat, bbox.min_lon, bbox.max_lon)

        # Quadtree tarama
        cells_to_process = deque([root])
        estimated_total = 100  # Tahmini ilerleme icin

        while cells_to_process and self.is_running:
            cell = cells_to_process.popleft()

            # Hucrenin polygon ile kesisimini kontrol et
            if not self._cell_intersects_polygon(cell, polygon):
                self.cells_skipped += 1
                continue

            # Hucreyi tara
            should_subdivide, new_parcels = self._scan_cell(cell, polygon)

            found_count += new_parcels
            self.cells_processed += 1

            # Progress guncelle (FAZ 1 icin %0-50)
            progress_pct = min(50, int((self.cells_processed / max(estimated_total, 1)) * 50))
            self.progress.emit(progress_pct, 100)

            if should_subdivide and cell.can_subdivide():
                # Hucreyi bol ve alt hucreleri kuyruğa ekle
                cell.subdivide()
                for child in cell.children:
                    cells_to_process.append(child)
                estimated_total += 3  # 1 hucre -> 4 hucre, net +3

            # Her 5 hucrede bir stats guncelle
            if self.cells_processed % 5 == 0:
                self.points_remaining = len(cells_to_process)
                self._emit_stats()

            time.sleep(self.delay)

        return found_count

    def _phase2_boundary_walk(self, polygon: List[Tuple[float, float]]) -> int:
        """FAZ 2: Sinir takibi - bulunan parsellerin kenarlarindan disa yuruyerek komsulari bul."""
        self.current_phase = "Boundary"
        found_count = 0

        if not self.found_parcels:
            self.log.emit("  Sinir takibi icin parsel yok, atlaniyor...")
            return 0

        # BFS kuyrugu - isle ve yeni bulunan parselleri de ekle
        parcels_to_walk = deque(self.found_parcels.copy())
        walked_parcels = set()  # Zaten yurunen parseller

        # Her parseli bir kez yuru
        iteration = 0

        while parcels_to_walk and self.is_running and iteration < self.BOUNDARY_WALK_MAX_ITERATIONS:
            iteration += 1
            current_batch = list(parcels_to_walk)
            parcels_to_walk.clear()
            new_in_iteration = 0

            self.log.emit(f"  Iterasyon {iteration}: {len(current_batch)} parsel siniri taranacak")

            for parcel_poly in current_batch:
                if not self.is_running:
                    break

                # Bos veya gecersiz poligonu atla
                if not parcel_poly or len(parcel_poly) < 3:
                    continue

                # Parsel hash'i (koordinat bazli - tuple hash ile)
                # Siralama baslangic noktasindan bagimsizlik saglar
                # round() kayan nokta hassasiyet farklarini onler
                poly_hash = tuple(sorted((round(lat, 5), round(lon, 5)) for lat, lon in parcel_poly))
                if poly_hash in walked_parcels:
                    continue
                walked_parcels.add(poly_hash)

                # Kenar noktalari cikar
                edge_points = self._extract_edge_points(parcel_poly, polygon)

                if not edge_points:
                    continue

                # Pruning uygula
                edge_points = self._prune_points(edge_points)

                if not edge_points:
                    continue

                # Batch sorgu - onceki/sonraki polygon sayisini karsilastir
                polys_before = len(self.found_parcels)
                new_parcels = self._query_and_process_points(edge_points)
                polys_after = len(self.found_parcels)

                found_count += new_parcels
                new_in_iteration += new_parcels
                self.boundary_walks += 1

                # Yeni eklenen polygonlari kuyruga ekle (geometrisi olan parseller)
                num_new_polys = polys_after - polys_before
                if num_new_polys > 0:
                    for poly in self.found_parcels[-num_new_polys:]:
                        parcels_to_walk.append(poly)

                time.sleep(self.delay)

            # Progress guncelle (FAZ 2 icin %50-75)
            progress_pct = 50 + min(25, iteration * 5)
            self.progress.emit(progress_pct, 100)

            self.log.emit(f"  Iterasyon {iteration} bitti: {new_in_iteration} yeni parsel")

            if new_in_iteration == 0:
                break  # Yeni parsel bulunamadi, sonraki faza gec

        return found_count

    def _phase3_gap_fill(self, polygon: List[Tuple[float, float]]) -> int:
        """FAZ 3: Kalan bosluklari daha yogun grid ile tara."""
        self.current_phase = "Gap Fill"
        found_count = 0

        # Daha yogun grid adimi
        dense_step = max(self.MIN_GAP_FILL_STEP, self.step_meters // self.GAP_FILL_STEP_DIVISOR)
        self.log.emit(f"  Yogun grid adimi: {dense_step}m")

        # Polygon bounding box
        bbox = KMLParser.get_bounding_box(polygon)
        center_lat = (bbox.min_lat + bbox.max_lat) / 2

        # Grid noktalari olustur
        lat_step = dense_step / 111000
        lon_step = dense_step / (111000 * math.cos(math.radians(center_lat)))

        all_points = []
        lat = bbox.min_lat
        while lat <= bbox.max_lat:
            lon = bbox.min_lon
            while lon <= bbox.max_lon:
                if KMLParser.point_in_polygon(lat, lon, polygon):
                    all_points.append((lat, lon))
                lon += lon_step
            lat += lat_step

        self.log.emit(f"  Toplam {len(all_points)} nokta olusturuldu")

        # Pruning - zaten bulunan parsellerin icindeki noktalari filtrele
        points = self._prune_points(all_points)
        self.gap_fill_points = len(points)
        self.log.emit(f"  Pruning sonrasi {len(points)} nokta kaldi ({len(all_points) - len(points)} elendi)")

        if not points:
            return 0

        # Batch'ler halinde sorgula
        remaining = deque(points)
        batch_num = 0
        total_batches = (len(points) + self.BATCH_SIZE - 1) // self.BATCH_SIZE

        while remaining and self.is_running:
            batch = [remaining.popleft() for _ in range(min(self.BATCH_SIZE, len(remaining)))]
            batch_num += 1

            new_parcels = self._query_and_process_points(batch)
            found_count += new_parcels

            # Progress guncelle (FAZ 3 icin %75-100)
            progress_pct = 75 + min(25, int((batch_num / max(total_batches, 1)) * 25))
            self.progress.emit(progress_pct, 100)

            if batch_num % 10 == 0:
                self.log.emit(f"  Batch {batch_num}/{total_batches} - {found_count} yeni parsel")
                self.points_remaining = len(remaining)
                self._emit_stats()

            time.sleep(self.delay)

        return found_count

    def _extract_edge_points(self, parcel_poly: List[Tuple[float, float]],
                              search_polygon: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Parsel kenarlarindan disa dogru noktalar cikarir."""
        edge_points = []
        n = len(parcel_poly)

        # Polygon kapali mi kontrol et (ilk nokta == son nokta)
        is_closed = (n >= 2 and parcel_poly[0] == parcel_poly[-1])
        # Kapali ise son tekrar eden noktayi atla, kapalI degilse tum kenarlari isle
        edge_count = n - 1 if is_closed else n

        for i in range(edge_count):
            p1 = parcel_poly[i]
            p2 = parcel_poly[(i + 1) % n]  # Son kenar icin ilk noktaya bagla

            # Kenar uzunlugu (metre)
            edge_lat = (p1[0] + p2[0]) / 2
            lat_diff = (p2[0] - p1[0]) * 111000
            lon_diff = (p2[1] - p1[1]) * 111000 * math.cos(math.radians(edge_lat))
            edge_length = math.sqrt(lat_diff**2 + lon_diff**2)

            if edge_length < self.MIN_EDGE_LENGTH_METERS:
                continue

            # Kenar boyunca noktalar
            num_points = max(1, int(edge_length / self.EDGE_STEP_METERS))

            for j in range(num_points + 1):
                # num_points her zaman >= 1 (max(1, ...) ile tanimli)
                t = j / num_points
                mid_lat = p1[0] + t * (p2[0] - p1[0])
                mid_lon = p1[1] + t * (p2[1] - p1[1])

                # Kenardan disa dogru normal vektor hesaplama
                # Kenar vektoru (metre): (lon_diff, lat_diff)
                # 90 derece saat yonunde dondurulmus normal: (lat_diff, -lon_diff)
                if edge_length > 0:
                    # Birim normal vektor bilesenleri
                    normal_x_unit = lat_diff / edge_length   # lon yonunde
                    normal_y_unit = -lon_diff / edge_length  # lat yonunde

                    # Kenarin her iki tarafinda da nokta olustur (ic/dis yon garantisi)
                    for direction in [1, -1]:
                        # Adim vektoru (metre)
                        delta_lat_m = direction * self.OUTWARD_STEP_METERS * normal_y_unit
                        delta_lon_m = direction * self.OUTWARD_STEP_METERS * normal_x_unit

                        # Metre -> derece donusumu
                        delta_lat_deg = delta_lat_m / 111000
                        delta_lon_deg = delta_lon_m / (111000 * math.cos(math.radians(mid_lat)))

                        new_lat = mid_lat + delta_lat_deg
                        new_lon = mid_lon + delta_lon_deg

                        # Sadece arama polygon'u icindeki noktalari ekle
                        if KMLParser.point_in_polygon(new_lat, new_lon, search_polygon):
                            edge_points.append((new_lat, new_lon))

        return edge_points

    def _query_and_process_points(self, points: List[Tuple[float, float]]) -> int:
        """Noktalari batch olarak sorgular ve sonuclari isler."""
        new_count = 0
        remaining = deque(points)

        while remaining and self.is_running:
            batch = [remaining.popleft() for _ in range(min(self.BATCH_SIZE, len(remaining)))]

            results, error = self._query_batch(batch)

            if error == 401:
                self.auth_error.emit()
                self.is_running = False
                return new_count

            if error or not results:
                continue

            for i, result in enumerate(results):
                if result and 'properties' in result:
                    _, is_new = self._process_found_parcel(result, batch[i])
                    if is_new:
                        new_count += 1

            time.sleep(self.delay)

        return new_count

    def _cell_intersects_polygon(self, cell: QuadTreeNode, polygon: List[Tuple[float, float]]) -> bool:
        """Hucrenin polygon ile kesisip kesismedigini kontrol eder."""
        # Hucrenin herhangi bir kosesi polygon icindeyse veya
        # Polygon'un herhangi bir kosesi hucre icindeyse kesisiyor demektir
        corners = cell.get_corners()
        for lat, lon in corners:
            if KMLParser.point_in_polygon(lat, lon, polygon):
                return True

        # Hucre merkezi polygon icinde mi?
        center = cell.get_center()
        if KMLParser.point_in_polygon(center[0], center[1], polygon):
            return True

        return False

    def _scan_cell(self, cell: QuadTreeNode, polygon: List[Tuple[float, float]]) -> Tuple[bool, int]:
        """
        Bir hucreyi tarar.
        Returns: (should_subdivide, new_parcel_count)
        """
        cell_size = cell.get_cell_size_meters()

        # Kucuk hucreler icin grid tarama yap
        if cell_size <= self.step_meters * 2:
            return self._grid_scan_cell(cell, polygon)

        # Buyuk hucreler icin sample noktalari sorgula
        sample_points = cell.get_sample_points()
        # Sadece polygon icindeki noktalari filtrele
        valid_points = [(lat, lon) for lat, lon in sample_points
                        if KMLParser.point_in_polygon(lat, lon, polygon)]

        if not valid_points:
            return False, 0

        # Zaten bulunan parsellerin icindeki noktalari filtrele (pruning)
        valid_points = self._prune_points(valid_points)

        if not valid_points:
            return False, 0

        # Batch sorgu
        results, error = self._query_batch(valid_points)

        if error == 401:
            self.auth_error.emit()
            self.is_running = False
            return False, 0

        if error or not results:
            return True, 0  # Hata durumunda subdivide et

        # Sonuclari isle
        new_count = 0
        unique_parcels = set()

        for i, result in enumerate(results):
            if result and 'properties' in result:
                parsel_id, is_new = self._process_found_parcel(result, valid_points[i])
                if is_new:
                    new_count += 1
                unique_parcels.add(parsel_id)

        # Karar ver: subdivide etmeli mi?
        # - Farkli parseller bulduysa -> subdivide (sinir bolgesi)
        # - Sadece 1 parsel ve hepsi ayni -> subdivide etme
        # - Hic parsel yok -> subdivide etme (bos alan)
        should_subdivide = len(unique_parcels) > 1 or (len(unique_parcels) == 1 and new_count > 0)

        return should_subdivide, new_count

    def _grid_scan_cell(self, cell: QuadTreeNode, polygon: List[Tuple[float, float]]) -> Tuple[bool, int]:
        """Kucuk hucreler icin yogun grid tarama."""
        points = self._generate_grid_for_cell(cell, polygon)
        points = self._prune_points(points)

        if not points:
            return False, 0

        new_count = 0

        # Noktalari batch'ler halinde isle
        remaining = deque(points)
        while remaining and self.is_running:
            batch = [remaining.popleft() for _ in range(min(self.BATCH_SIZE, len(remaining)))]

            results, error = self._query_batch(batch)

            if error == 401:
                self.auth_error.emit()
                self.is_running = False
                return False, new_count

            if error or not results:
                continue

            for i, result in enumerate(results):
                if result and 'properties' in result:
                    _, is_new = self._process_found_parcel(result, batch[i])
                    if is_new:
                        new_count += 1

            time.sleep(self.delay)

        return False, new_count

    def _generate_grid_for_cell(self, cell: QuadTreeNode,
                                  polygon: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Hucre icin grid noktalari olusturur."""
        center_lat = (cell.min_lat + cell.max_lat) / 2
        lat_step = self.step_meters / 111000
        lon_step = self.step_meters / (111000 * math.cos(math.radians(center_lat)))

        points = []
        lat = cell.min_lat
        while lat <= cell.max_lat:
            lon = cell.min_lon
            while lon <= cell.max_lon:
                if KMLParser.point_in_polygon(lat, lon, polygon):
                    points.append((lat, lon))
                lon += lon_step
            lat += lat_step
        return points

    def _prune_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Zaten bulunan parsellerin icindeki noktalari filtreler."""
        if not self.found_parcels:
            return points

        return [p for p in points
                if not any(KMLParser.point_in_polygon(p[0], p[1], poly) for poly in self.found_parcels)]

    def _query_batch(self, points: List[Tuple[float, float]]) -> Tuple[list, any]:
        """Batch sorgu yapar ve hatalari yonetir."""
        self.api_calls += 1
        self.total_points_queried += len(points)
        max_retries = 2

        for attempt in range(max_retries + 1):
            results, error = self.client.get_batch(points)

            if error == 401:
                return None, 401

            if not error:
                # Null/bos sonuclari say
                if results:
                    for r in results:
                        if r is None or (isinstance(r, dict) and 'properties' not in r):
                            self.null_responses += 1
                return results, None

            if attempt < max_retries:
                self.log.emit(f"  Batch hatasi, tekrar deneniyor ({attempt + 1}/{max_retries})...")
                time.sleep(self.delay * 2)

        # Tum batch sorgulanamadi (ag hatasi vs)
        self.failed_points += len(points)
        return None, error

    def _log_summary(self, found_count: int):
        """Tarama ozeti loglar."""
        self.log.emit(f"\n{'='*50}")
        self.log.emit("TARAMA OZETI")
        self.log.emit("="*50)
        self.log.emit(f"Toplam parsel: {found_count}")
        self.log.emit(f"API cagrilari: {self.api_calls} batch ({self.total_points_queried} nokta)")
        self.log.emit(f"Parsel bulunamadi: {self.null_responses} nokta")
        if self.failed_points > 0:
            self.log.emit(f"Sorgu hatasi (ag vs): {self.failed_points} nokta")
        self.log.emit(f"Quadtree hucreleri: {self.cells_processed} islendi, {self.cells_skipped} atlandi")
        self.log.emit(f"Sinir takibi: {self.boundary_walks} parsel siniri tarandi")
        self.log.emit(f"Bosluk doldurma: {self.gap_fill_points} nokta tarandi")
        self.log.emit("Hibrit algoritma (Quadtree + Boundary + Gap Fill) tamamlandi")

    def stop(self):
        """Taramayi durdurur."""
        self.is_running = False


# ==================== ANA PENCERE ====================

class MainWindow(QMainWindow):
    """Ana uygulama penceresi."""

    def __init__(self):
        """Ana pencereyi olusturur ve yapilandirir."""
        super().__init__()
        self.setWindowTitle("TKGM Parsel Veri Cekme")
        self.setMinimumSize(800, 700)

        # Ayarlari yukle
        self.settings = QSettings("TKGM", "ParselCekme")
        self.polygons = []
        self.parcels = {}
        self.worker = None

        # File logging setup
        self.setup_logging()

        # Tema
        setTheme(Theme.LIGHT)

        self.setup_ui()
        self.load_settings()

    def setup_logging(self):
        """Log dosyasi olusturur."""
        log_dir = Path.home() / ".tkgm_parsel"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        self.file_logger = logging.getLogger("tkgm_scan")
        self.file_logger.setLevel(logging.DEBUG)

        handler = logging.FileHandler(log_file, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.file_logger.addHandler(handler)

        self.log_file_path = log_file

    def setup_ui(self):
        """Kullanici arayuzunu olusturur."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Baslik
        title = TitleLabel("TKGM Parsel Veri Cekme")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Worker URL Card
        url_card = CardWidget()
        url_layout = QVBoxLayout(url_card)
        url_layout.addWidget(SubtitleLabel("Cloudflare Worker URL"))
        self.worker_url_input = LineEdit()
        self.worker_url_input.setPlaceholderText("https://your-worker.workers.dev")
        url_layout.addWidget(self.worker_url_input)

        # API Key (Opsiyonel)
        url_layout.addWidget(BodyLabel("API Key (Opsiyonel)"))
        self.api_key_input = LineEdit()
        self.api_key_input.setPlaceholderText("Bos birakilabilir - Worker herkese acik olur")
        url_layout.addWidget(self.api_key_input)

        layout.addWidget(url_card)

        # KML Card
        kml_card = CardWidget()
        kml_layout = QVBoxLayout(kml_card)
        kml_layout.addWidget(SubtitleLabel("KML Dosyasi"))

        kml_row = QHBoxLayout()
        self.kml_path_input = LineEdit()
        self.kml_path_input.setPlaceholderText("KML dosyasi secin...")
        self.kml_path_input.setReadOnly(True)
        kml_row.addWidget(self.kml_path_input)

        kml_btn = PushButton("Sec")
        kml_btn.setIcon(FluentIcon.FOLDER)
        kml_btn.clicked.connect(self.select_kml)
        kml_row.addWidget(kml_btn)
        kml_layout.addLayout(kml_row)

        self.kml_info = BodyLabel("")
        kml_layout.addWidget(self.kml_info)
        layout.addWidget(kml_card)

        # Ayarlar Card
        settings_card = CardWidget()
        settings_layout = QHBoxLayout(settings_card)

        # Grid araligi
        grid_layout = QVBoxLayout()
        grid_layout.addWidget(BodyLabel("Grid Araligi (metre)"))
        self.step_spin = SpinBox()
        self.step_spin.setRange(10, 200)
        self.step_spin.setValue(30)
        grid_layout.addWidget(self.step_spin)
        settings_layout.addLayout(grid_layout)

        # Bekleme
        delay_layout = QVBoxLayout()
        delay_layout.addWidget(BodyLabel("Bekleme Suresi (sn)"))
        self.delay_spin = DoubleSpinBox()
        self.delay_spin.setRange(0.1, 5.0)
        self.delay_spin.setValue(0.3)
        self.delay_spin.setSingleStep(0.1)
        delay_layout.addWidget(self.delay_spin)
        settings_layout.addLayout(delay_layout)

        layout.addWidget(settings_card)

        # Butonlar
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.start_btn = PrimaryPushButton("Taramayi Baslat")
        self.start_btn.setIcon(FluentIcon.PLAY)
        self.start_btn.clicked.connect(self.start_scan)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = PushButton("Durdur")
        self.stop_btn.setIcon(FluentIcon.PAUSE)
        self.stop_btn.clicked.connect(self.stop_scan)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        self.save_btn = PushButton("Kaydet")
        self.save_btn.setIcon(FluentIcon.SAVE)
        self.save_btn.clicked.connect(self.save_results)
        btn_layout.addWidget(self.save_btn)

        self.refine_btn = PushButton("Detaylandir")
        self.refine_btn.setIcon(FluentIcon.SYNC)
        self.refine_btn.clicked.connect(self.refine_scan)
        self.refine_btn.setEnabled(False)
        self.refine_btn.setToolTip("Eksik bolgeler icin daha yogun grid ile tekrar tara")
        btn_layout.addWidget(self.refine_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Progress
        progress_card = CardWidget()
        progress_layout = QVBoxLayout(progress_card)

        self.status_label = BodyLabel("Hazir")
        progress_layout.addWidget(self.status_label)

        self.progress_bar = ProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        # Detayli istatistik satiri
        self.stats_label = BodyLabel("")
        self.stats_label.setStyleSheet("color: #666; font-size: 11px;")
        progress_layout.addWidget(self.stats_label)

        layout.addWidget(progress_card)

        # Log
        log_card = CardWidget()
        log_layout = QVBoxLayout(log_card)
        log_layout.addWidget(SubtitleLabel("Log"))

        self.log_text = TextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_card)

    def load_settings(self):
        """Kaydedilmis ayarlari yukler."""
        url = self.settings.value("worker_url", "")
        if url:
            self.worker_url_input.setText(url)

        api_key = self.settings.value("api_key", "")
        if api_key:
            self.api_key_input.setText(api_key)

        step = self.settings.value("step_meters", 30, type=int)
        self.step_spin.setValue(step)

        delay = self.settings.value("delay", 0.3, type=float)
        self.delay_spin.setValue(delay)

    def save_settings(self):
        """Mevcut ayarlari kaydeder."""
        self.settings.setValue("worker_url", self.worker_url_input.text())
        self.settings.setValue("api_key", self.api_key_input.text())
        self.settings.setValue("step_meters", self.step_spin.value())
        self.settings.setValue("delay", self.delay_spin.value())

    def log(self, message: str):
        """Log alanina zaman damgali mesaj ekler ve dosyaya yazar."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # Dosyaya da yaz
        self.file_logger.info(message)

    def select_kml(self):
        """KML dosyasi secme dialogunu acar."""
        path, _ = QFileDialog.getOpenFileName(
            self, "KML Dosyasi Sec", "",
            "KML Files (*.kml);;All Files (*)"
        )
        if path:
            self.kml_path_input.setText(path)
            self.load_kml(path)

    def load_kml(self, path: str):
        """KML dosyasini yukler ve parse eder."""
        try:
            self.polygons = KMLParser.parse(path)
            info = f"{len(self.polygons)} polygon bulundu"

            total_points = 0
            for poly in self.polygons:
                bbox = KMLParser.get_bounding_box(poly)
                center_lat = (bbox.min_lat + bbox.max_lat) / 2
                lat_step = self.step_spin.value() / 111000
                lon_step = self.step_spin.value() / (111000 * math.cos(math.radians(center_lat)))
                lat_count = int((bbox.max_lat - bbox.min_lat) / lat_step) + 1
                lon_count = int((bbox.max_lon - bbox.min_lon) / lon_step) + 1
                total_points += lat_count * lon_count

            info += f" (~{total_points} tarama noktasi)"
            self.kml_info.setText(info)
            self.log(f"KML yuklendi: {path}")
            self.log(info)

            InfoBar.success(
                title="Basarili",
                content=info,
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000
            )
        except (ET.ParseError, ValueError, IOError) as e:
            self.log(f"HATA: {e}")
            InfoBar.error(
                title="Hata",
                content=str(e),
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=5000
            )

    def start_scan(self):
        """Tarama islemini baslatir."""
        if not self.polygons:
            InfoBar.warning(
                title="Uyari",
                content="Lutfen once KML dosyasi secin",
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000
            )
            return

        url = self.worker_url_input.text().strip()
        if not url or "your-worker" in url:
            InfoBar.warning(
                title="Uyari",
                content="Lutfen Worker URL girin",
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000
            )
            return

        self.save_settings()
        self.parcels = {}  # Yeni tarama, onceki parselleri temizle

        # API Key uyarisi
        api_key = self.api_key_input.text().strip()
        if not api_key:
            InfoBar.info(
                title="Bilgi",
                content="API Key girilmedi - Worker herkese acik",
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000
            )

        self._start_worker(self.step_spin.value(), reuse_parcels=False)
        self.log("Tarama baslatildi...")

    def _start_worker(self, step_meters: int, reuse_parcels: bool = False):
        """Worker olusturur ve baslatir (ortak mantik)."""
        url = self.worker_url_input.text().strip()
        api_key = self.api_key_input.text().strip() or None
        client = TKGMClient(url, api_key)

        self.worker = ScanWorker(
            client,
            self.polygons,
            step_meters,
            self.delay_spin.value()
        )

        # Eger onceki parselleri koruyacaksak, worker'a aktar
        if reuse_parcels and self.parcels:
            for parsel_id, data in self.parcels.items():
                self.worker.found_ids.add(parsel_id)
                if 'geometry' in data and data['geometry'].get('type') == 'Polygon':
                    coord_list = data['geometry'].get('coordinates') or []
                    if coord_list and len(coord_list[0]) >= 3:
                        self.worker.found_parcels.append(ScanWorker.geojson_coords_to_poly(coord_list[0]))

        # Sinyalleri bagla
        self.worker.progress.connect(self.on_progress)
        self.worker.stats_update.connect(self.on_stats_update)
        self.worker.log.connect(self.log)
        self.worker.parcel_found.connect(self.on_parcel_found)
        self.worker.finished_scan.connect(self.on_finished)
        self.worker.auth_error.connect(self.on_auth_error)
        self.worker.start()

        # Buton durumlarini guncelle
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.refine_btn.setEnabled(False)

    def stop_scan(self):
        """Devam eden taramayi durdurur."""
        if self.worker:
            self.worker.stop()
            self.log("Durdurma istegi gonderildi...")

    def on_progress(self, current: int, total: int):
        """Ilerleme sinyalini isler."""
        percent = int(current / total * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)
        self.status_label.setText(f"Taraniyor: %{percent} | Bulunan: {len(self.parcels)} parsel")

    def on_stats_update(self, stats: dict):
        """Detayli istatistik sinyalini isler."""
        phase = stats.get("phase", "")
        queried = stats.get("queried", 0)
        remaining = stats.get("remaining", 0)
        null_resp = stats.get("null_responses", 0)
        failed = stats.get("failed_points", 0)
        failed_str = f" | Hata: {failed}" if failed > 0 else ""
        self.stats_label.setText(
            f"Faz: {phase} | Sorgulanan: {queried} | Kalan: {remaining} | Bos: {null_resp}{failed_str}"
        )

    def on_parcel_found(self, parsel_id: str, data: dict):
        """Bulunan parseli kaydeder."""
        self.parcels[parsel_id] = data

    def on_auth_error(self):
        """API Key hatasini isler."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Hata - API Key hatali")

        InfoBar.error(
            title="API Key Hatasi",
            content="API Key hatali veya eksik. Worker'da API_KEY tanimliysa ayni key'i girin.",
            parent=self,
            position=InfoBarPosition.TOP_RIGHT,
            duration=8000
        )

    def on_finished(self, count: int):
        """Tarama tamamlandiginda cagrilir."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.refine_btn.setEnabled(True)  # Detaylandir butonunu aktif et
        self.status_label.setText(f"Tamamlandi - {count} parsel bulundu")
        self.log(f"Tarama tamamlandi! {count} benzersiz parsel bulundu")
        self.log(f"Log dosyasi: {self.log_file_path}")

        InfoBar.success(
            title="Tamamlandi",
            content=f"{count} parsel bulundu. Eksik bolge varsa 'Detaylandir' butonuna basin.",
            parent=self,
            position=InfoBarPosition.TOP_RIGHT,
            duration=5000
        )

    def refine_scan(self):
        """Eksik bolgeleri daha yogun grid ile tekrar tarar."""
        if not self.polygons:
            InfoBar.warning(
                title="Uyari",
                content="Taranacak polygon yok",
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000
            )
            return

        url = self.worker_url_input.text().strip()
        if not url:
            return

        # Daha yogun grid (mevcut step'in 1/3'u)
        refined_step = max(
            ScanWorker.MIN_REFINE_STEP,
            self.step_spin.value() // ScanWorker.REFINE_STEP_DIVISOR
        )

        self.log("\n=== DETAYLANDIRMA TARAMASI ===")
        self.log(f"Yogun grid adimi: {refined_step}m (onceki: {self.step_spin.value()}m)")

        self._start_worker(refined_step, reuse_parcels=True)
        self.log(f"Detaylandirma baslatildi ({len(self.parcels)} mevcut parsel korunuyor)...")

    def save_results(self):
        """Sonuclari dosyaya kaydetme dialogunu acar."""
        if not self.parcels:
            InfoBar.warning(
                title="Uyari",
                content="Kaydedilecek veri yok",
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Kaydet", "",
            "KML (*.kml);;GeoJSON (*.geojson);;JSON (*.json)"
        )

        if not path:
            return

        try:
            if path.endswith('.kml'):
                self.save_kml(path)
            else:
                self.save_geojson(path)

            self.log(f"Kaydedildi: {path}")
            InfoBar.success(
                title="Kaydedildi",
                content=path,
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000
            )
        except (IOError, OSError) as e:
            InfoBar.error(
                title="Hata",
                content=str(e),
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=5000
            )

    def save_geojson(self, path: str):
        """Parselleri GeoJSON formatinda kaydeder."""
        geojson = {
            "type": "FeatureCollection",
            "features": list(self.parcels.values())
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

    def save_kml(self, path: str):
        """Parselleri KML formatinda kaydeder."""
        kml = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>TKGM Parseller</name>
<Style id="s"><LineStyle><color>ff0000ff</color><width>2</width></LineStyle><PolyStyle><color>4d0000ff</color></PolyStyle></Style>
'''
        for pid, p in self.parcels.items():
            props = p.get('properties', {})
            geom = p.get('geometry', {})
            if geom.get('type') != 'Polygon':
                continue
            coords = geom.get('coordinates', [[]])[0]
            coord_str = ' '.join([f"{c[0]},{c[1]},0" for c in coords])
            name = props.get('ozet', pid)

            # Detayli aciklama
            desc = f"""<![CDATA[
<b>Il:</b> {props.get('ilAd', '-')}<br/>
<b>Ilce:</b> {props.get('ilceAd', '-')}<br/>
<b>Mahalle:</b> {props.get('mahalleAd', '-')}<br/>
<b>Mevkii:</b> {props.get('mevkii', '-')}<br/>
<b>Ada No:</b> {props.get('adaNo', '-')}<br/>
<b>Parsel No:</b> {props.get('parselNo', '-')}<br/>
<b>Pafta:</b> {props.get('pafta', '-')}<br/>
<b>Alan:</b> {props.get('alan', '-')} m²<br/>
<b>Nitelik:</b> {props.get('nitelik', '-')}<br/>
<b>Durum:</b> {props.get('zeminKmdurum', '-')}
]]>"""

            kml += (f'<Placemark><name>{name}</name><description>{desc}</description><styleUrl>#s</styleUrl>'
                    f'<Polygon><outerBoundaryIs><LinearRing><coordinates>{coord_str}</coordinates>'
                    '</LinearRing></outerBoundaryIs></Polygon></Placemark>\n')

        kml += '</Document></kml>'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(kml)

    def closeEvent(self, event):  # pylint: disable=invalid-name
        """Pencere kapatilirken ayarlari kaydeder ve worker'i durdurur."""
        self.save_settings()
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        event.accept()


# ==================== MAIN ====================

def main():
    """Uygulamayi baslatir."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
