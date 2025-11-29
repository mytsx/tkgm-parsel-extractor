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
from typing import List, Tuple, Set, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QButtonGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings

# Fluent Widgets
from qfluentwidgets import (
    PushButton, PrimaryPushButton, LineEdit, BodyLabel, TitleLabel,
    SubtitleLabel, CardWidget, ProgressBar, TextEdit, RadioButton,
    InfoBar, InfoBarPosition, FluentIcon, setTheme, Theme
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
    """
    Cloudflare Worker uzerinden TKGM API istemcisi
    - Multi-worker pool desteği (workers.json'dan okur)
    - Hedged requests (paralel deneme)
    - Worker health tracking (cooldown sistemi)
    - Akıllı paralel sayısı (worker sayısına göre)
    """

    REQUEST_TIMEOUT = 15       # Tekil istek timeout (saniye)
    BATCH_REQUEST_TIMEOUT = 60  # Batch istek timeout (saniye)
    COOLDOWN_SECONDS = 300     # 403 alan worker 5 dk cooldown
    CONFIG_FILE = "workers.json"

    @classmethod
    def from_config(cls, config_path: str = None) -> 'TKGMClient':
        """workers.json dosyasından client oluşturur."""
        if config_path is None:
            # Uygulama dizininde ara
            config_path = Path(__file__).parent / cls.CONFIG_FILE

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Worker config bulunamadı: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        workers = config.get('workers', [])
        api_key = config.get('api_key', '')

        if not workers:
            raise ValueError("workers.json'da worker tanımlı değil")

        return cls(workers=workers, api_key=api_key)

    def __init__(self, workers: list = None, api_key: str = None, worker_url: str = None):
        """
        Worker listesi veya URL ile istemci oluşturur.
        workers: ["url1", "url2", ...] listesi (önerilen)
        worker_url: Tek URL veya virgülle ayrılmış URL'ler (geriye uyumluluk)
        """
        # Worker listesini belirle
        if workers:
            self.worker_urls = [u.rstrip('/') for u in workers]
        elif worker_url:
            urls = [u.strip().rstrip('/') for u in worker_url.split(',') if u.strip()]
            self.worker_urls = urls if urls else [worker_url.rstrip('/')]
        else:
            raise ValueError("workers veya worker_url gerekli")

        self.worker_url = self.worker_urls[0]  # Geriye uyumluluk
        self.current_worker_idx = 0
        self.api_key = api_key

        # Akıllı paralel sayısı: worker sayısına göre
        # 1 worker -> 1, 2-3 worker -> 2, 4+ worker -> 3
        n = len(self.worker_urls)
        self.parallel_tries = min(3, max(1, (n + 1) // 2)) if n < 6 else 3

        # Worker health tracking
        self.worker_cooldown: dict = {}  # {url: cooldown_end_time}
        self.worker_success: dict = {url: 0 for url in self.worker_urls}
        self.worker_fail: dict = {url: 0 for url in self.worker_urls}

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        })
        if api_key:
            self.session.headers["X-API-Key"] = api_key

    def _get_healthy_workers(self, count: int) -> List[str]:
        """Sağlıklı worker'lardan count kadar döndürür (round-robin)."""
        now = time.time()
        healthy = []
        checked = 0

        while len(healthy) < count and checked < len(self.worker_urls) * 2:
            url = self.worker_urls[self.current_worker_idx]
            self.current_worker_idx = (self.current_worker_idx + 1) % len(self.worker_urls)
            checked += 1

            # Cooldown kontrolü
            cooldown_end = self.worker_cooldown.get(url, 0)
            if now >= cooldown_end:
                healthy.append(url)

        # Eğer yeterli sağlıklı worker yoksa, cooldown'dakileri de ekle
        if len(healthy) < count:
            for url in self.worker_urls:
                if url not in healthy:
                    healthy.append(url)
                if len(healthy) >= count:
                    break

        return healthy[:count]

    def _mark_worker_success(self, url: str):
        """Worker başarılı - cooldown'dan çıkar."""
        self.worker_success[url] = self.worker_success.get(url, 0) + 1
        if url in self.worker_cooldown:
            del self.worker_cooldown[url]

    def _mark_worker_fail(self, url: str, is_rate_limit: bool = False):
        """Worker başarısız - rate limit ise cooldown'a al."""
        self.worker_fail[url] = self.worker_fail.get(url, 0) + 1
        if is_rate_limit:
            self.worker_cooldown[url] = time.time() + self.COOLDOWN_SECONDS

    def get_worker_count(self) -> int:
        """Toplam worker sayısını döndürür."""
        return len(self.worker_urls)

    def get_healthy_worker_count(self) -> int:
        """Sağlıklı (cooldown'da olmayan) worker sayısını döndürür."""
        now = time.time()
        return sum(1 for url in self.worker_urls if now >= self.worker_cooldown.get(url, 0))

    def get_worker_stats(self) -> dict:
        """Worker istatistiklerini döndürür."""
        now = time.time()
        return {
            "total": len(self.worker_urls),
            "healthy": self.get_healthy_worker_count(),
            "cooldown": len(self.worker_urls) - self.get_healthy_worker_count(),
            "success_total": sum(self.worker_success.values()),
            "fail_total": sum(self.worker_fail.values())
        }

    def _request_single_worker(self, worker: str, url_path: str, method: str = "GET", payload: dict = None) -> tuple:
        """Tek worker'a istek at. Returns (response_data, status_code, is_success)"""
        try:
            full_url = f"{worker}{url_path}"
            timeout = self.BATCH_REQUEST_TIMEOUT if method == "POST" else self.REQUEST_TIMEOUT

            if method == "POST":
                response = self.session.post(full_url, json=payload, timeout=timeout)
            else:
                response = self.session.get(full_url, timeout=timeout)

            return response, response.status_code, response.status_code == 200
        except requests.Timeout:
            return None, "TIMEOUT", False
        except requests.RequestException as e:
            return None, str(e), False

    def _hedged_request(self, url_path: str, method: str = "GET", payload: dict = None) -> tuple:
        """
        Hedged request - birden fazla worker'a paralel istek at, ilk başarılıyı döndür.
        Returns (response, worker_url, status_code)
        """
        workers = self._get_healthy_workers(self.parallel_tries)

        with ThreadPoolExecutor(max_workers=self.parallel_tries) as executor:
            futures = {
                executor.submit(self._request_single_worker, w, url_path, method, payload): w
                for w in workers
            }

            for future in as_completed(futures):
                worker = futures[future]
                try:
                    response, status_code, is_success = future.result()

                    if is_success:
                        self._mark_worker_success(worker)
                        # Diğer futures'ları iptal et
                        for f in futures:
                            f.cancel()
                        return response, worker, status_code

                    # Rate limit (403) - cooldown'a al
                    if status_code == 403:
                        self._mark_worker_fail(worker, is_rate_limit=True)
                    else:
                        self._mark_worker_fail(worker, is_rate_limit=False)

                except Exception:
                    self._mark_worker_fail(worker, is_rate_limit=False)

        # Hiçbiri başarılı olmadı
        return None, None, 403

    def get_parsel(self, lat: float, lon: float) -> tuple:
        """Returns (data, error_code) tuple - hedged request kullanır"""
        response, worker, status_code = self._hedged_request(f"/parsel/{lat}/{lon}")

        if response and status_code == 200:
            try:
                data = response.json()
                if 'properties' in data:
                    return data, None
            except Exception:
                pass

        if status_code == 401:
            return None, 401
        return None, status_code

    def get_batch(self, coordinates: list) -> tuple:
        """
        Batch sorgu - hedged request ile birden fazla worker dener
        Args:
            coordinates: [(lat, lon), (lat, lon), ...] listesi
        Returns:
            (results_list, meta_dict, error_code) tuple
            results_list: [{data}, {data}, ...] veya None'lar
            meta_dict: {"found": N, "empty": M, "error": K, "error_coords": [...]}
        """
        payload = {"coordinates": [{"lat": c[0], "lon": c[1]} for c in coordinates]}
        response, worker, status_code = self._hedged_request("/batch", method="POST", payload=payload)

        if not response or status_code != 200:
            if status_code == 401:
                return None, None, 401
            if status_code == 429:
                return None, {"rate_limited": True}, 429
            return None, None, status_code

        try:
            data = response.json()
            results = data.get('results', [])
            stats = data.get('stats', {})

            # Yeni format kontrolu
            is_new_format = results and isinstance(results[0], dict) and 'status' in results[0]

            processed = []
            error_coords = []

            if is_new_format:
                for r in results:
                    r_status = r.get('status')
                    if r_status == 'found':
                        processed.append(r.get('data'))
                    elif r_status == 'error':
                        processed.append(None)
                        coord = r.get('coord', {})
                        error_coords.append((coord.get('lat'), coord.get('lon')))
                    else:
                        processed.append(None)

                meta = {
                    "found": stats.get('found', 0),
                    "empty": stats.get('empty', 0),
                    "error": stats.get('error', 0),
                    "error_coords": error_coords
                }
            else:
                for r in results:
                    if r and isinstance(r, dict) and 'properties' in r:
                        processed.append(r)
                    else:
                        processed.append(None)

                found_count = len([p for p in processed if p])
                meta = {
                    "found": found_count,
                    "empty": len(processed) - found_count,
                    "error": 0,
                    "error_coords": []
                }

            return processed, meta, None
        except Exception as e:
            return None, {"error": str(e)}, "PARSE_ERROR"


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


# ==================== TARAMA KALİTESİ ====================

class ScanQuality(Enum):
    """Tarama kalite seviyeleri."""
    FAST = "fast"           # Hizli - buyuk alanlar icin
    BALANCED = "balanced"   # Dengeli - onerilen
    DETAILED = "detailed"   # Detayli - kucuk parseller icin


# ==================== SMART SCAN WORKER ====================

class ScanWorker(QThread):
    """
    Akilli tarama algoritmasi:
    - Global queried_coords seti (ayni nokta 2 kez sorgulanmaz)
    - Paralel batch istekleri (5-10x hiz artisi)
    - Adaptif grid (kaba -> yogun)
    - Otomatik parametre secimi
    """

    # Batch boyutu (Worker limiti 20)
    BATCH_SIZE = 15

    # Paralel istek sayisi (rate limit icin dusuk tutuldu)
    PARALLEL_BATCHES = 2

    # Batch arasi bekleme suresi (ms)
    BATCH_DELAY_MS = 200

    # Kalite seviyeleri: (kaba_grid, yogun_grid, boundary_step)
    QUALITY_SETTINGS = {
        ScanQuality.FAST: (100, 40, 15),
        ScanQuality.BALANCED: (60, 20, 10),
        ScanQuality.DETAILED: (40, 10, 5),
    }

    # Koordinat yuvarlamasi (tekrar sorgu onleme icin)
    COORD_PRECISION = 6  # ~11cm hassasiyet

    # Sinyaller
    progress = pyqtSignal(int, int)
    stats_update = pyqtSignal(dict)
    log = pyqtSignal(str)
    parcel_found = pyqtSignal(str, dict)
    finished_scan = pyqtSignal(int)
    auth_error = pyqtSignal()

    def __init__(self, client: TKGMClient, polygons: List, quality: ScanQuality = ScanQuality.BALANCED):
        """Akilli tarama worker'i olusturur."""
        super().__init__()
        self.client = client
        self.polygons = polygons
        self.quality = quality
        self.is_running = True

        # Kalite ayarlarini al
        self.coarse_step, self.dense_step, self.boundary_step = self.QUALITY_SETTINGS[quality]

        # *** GLOBAL SORGULANAN KOORDİNATLAR - ASLA TEKRAR SORGULANMAZ ***
        self.queried_coords: Set[Tuple[int, int]] = set()

        # Bulunan parseller
        self.found_ids: Set[str] = set()
        self.found_parcels: List[List[Tuple[float, float]]] = []  # Geometriler

        # Istatistikler
        self.api_calls = 0
        self.total_points_queried = 0
        self.duplicate_skipped = 0  # Tekrar sorgu engellenen
        self.points_remaining = 0
        self.current_phase = ""
        self.null_responses = 0
        self.failed_points = 0

    def _coord_key(self, lat: float, lon: float) -> Tuple[int, int]:
        """Koordinati hash key'e cevirir (yuvarlanmis integer)."""
        # 6 decimal = ~11cm hassasiyet
        return (round(lat * 10**self.COORD_PRECISION),
                round(lon * 10**self.COORD_PRECISION))

    def _filter_new_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Daha once sorgulanmamis noktalari filtreler."""
        new_points = []
        for lat, lon in points:
            key = self._coord_key(lat, lon)
            if key not in self.queried_coords:
                new_points.append((lat, lon))
        return new_points

    def _mark_as_queried(self, points: List[Tuple[float, float]]):
        """Noktalari sorgulandi olarak isaretler."""
        for lat, lon in points:
            self.queried_coords.add(self._coord_key(lat, lon))

    @staticmethod
    def geojson_coords_to_poly(coords: list) -> List[Tuple[float, float]]:
        """GeoJSON koordinatlarini (lon, lat) -> (lat, lon) tuple listesine donusturur."""
        return [(c[1], c[0]) for c in coords]

    def _process_result(self, result: dict, point: Tuple[float, float]) -> bool:
        """Sonucu isle, yeni parsel mi dondur."""
        if not result or 'properties' not in result:
            return False

        props = result['properties']
        parsel_id = props.get('ozet', f"{point[0]:.6f}_{point[1]:.6f}")

        if parsel_id in self.found_ids:
            return False

        self.found_ids.add(parsel_id)
        self.parcel_found.emit(parsel_id, result)
        self.log.emit(f"  ✓ {parsel_id} - {props.get('nitelik', '')} ({props.get('alan', '')})")

        # Geometriyi kaydet
        if 'geometry' in result and result['geometry'].get('type') == 'Polygon':
            coords = result['geometry'].get('coordinates', [[]])
            if coords and len(coords[0]) >= 3:
                self.found_parcels.append(self.geojson_coords_to_poly(coords[0]))

        return True

    def _emit_stats(self):
        """UI'a istatistik gonder."""
        self.stats_update.emit({
            "phase": self.current_phase,
            "found": len(self.found_ids),
            "api_calls": self.api_calls,
            "queried": self.total_points_queried,
            "remaining": self.points_remaining,
            "skipped": self.duplicate_skipped,
            "null_responses": self.null_responses,
            "failed_points": self.failed_points
        })

    def _query_batch_single(self, points: List[Tuple[float, float]]) -> Tuple[List, dict, Optional[any]]:
        """Tek batch sorgusu (paralel calisma icin)."""
        try:
            results, meta, error = self.client.get_batch(points)
            if error == 401:
                return [], {}, 401
            if error:
                return [], meta or {}, error
            return results or [], meta or {}, None
        except Exception as e:
            return [], {"exception": str(e)}, str(e)

    def _query_parallel(self, all_points: List[Tuple[float, float]]) -> int:
        """Paralel batch sorgulari yapar - retry destekli."""
        if not all_points:
            return 0

        # Tekrar sorgulanacak noktalari filtrele
        new_points = self._filter_new_points(all_points)
        self.duplicate_skipped += len(all_points) - len(new_points)

        if not new_points:
            return 0

        # Batch'lere bol
        batches = []
        for i in range(0, len(new_points), self.BATCH_SIZE):
            batches.append(new_points[i:i + self.BATCH_SIZE])

        new_count = 0
        total_batches = len(batches)
        processed = 0
        retry_queue = []  # Basarisiz batch'ler icin

        # Paralel istek gonder
        with ThreadPoolExecutor(max_workers=self.PARALLEL_BATCHES) as executor:
            future_to_batch = {}
            batch_idx = 0
            last_submit_time = 0

            while batch_idx < len(batches) or future_to_batch:
                if not self.is_running:
                    break

                # Yeni batch'ler ekle (max PARALLEL_BATCHES kadar)
                # Her batch arasinda delay (rate limit onleme)
                while batch_idx < len(batches) and len(future_to_batch) < self.PARALLEL_BATCHES:
                    # BATCH_DELAY_MS delay between submissions
                    delay_sec = self.BATCH_DELAY_MS / 1000
                    current_time = time.time()
                    if current_time - last_submit_time < delay_sec:
                        time.sleep(delay_sec - (current_time - last_submit_time))

                    batch = batches[batch_idx]
                    self._mark_as_queried(batch)
                    future = executor.submit(self._query_batch_single, batch)
                    future_to_batch[future] = (batch_idx, batch, 0)  # 0 = retry count
                    batch_idx += 1
                    last_submit_time = time.time()
                    self.api_calls += 1
                    self.total_points_queried += len(batch)

                # Tamamlanan future'lari isle
                done_futures = [f for f in future_to_batch if f.done()]

                for future in done_futures:
                    _, batch, retry_count = future_to_batch.pop(future)
                    processed += 1

                    try:
                        results, meta, error = future.result()

                        if error == 401:
                            self.auth_error.emit()
                            self.is_running = False
                            return new_count

                        # Rate limit veya timeout - retry'a ekle
                        if error == 429 or error == "TIMEOUT":
                            if retry_count < 3:
                                retry_queue.append((batch, retry_count + 1))
                                self.log.emit(f"  ⚠ Rate limit/timeout, retry kuyruğuna eklendi (deneme {retry_count + 1})")
                            else:
                                self.failed_points += len(batch)
                            continue

                        if error:
                            self.failed_points += len(batch)
                            continue

                        # Worker'dan gelen hata koordinatlarini retry'a ekle
                        error_coords = meta.get('error_coords', []) if meta else []
                        if error_coords and retry_count < 3:
                            retry_queue.append((error_coords, retry_count + 1))
                            self.log.emit(f"  ⚠ {len(error_coords)} koordinat hatali, retry'a eklendi")

                        # Sonuclari isle
                        for i, result in enumerate(results):
                            if result and 'properties' in result:
                                if self._process_result(result, batch[i]):
                                    new_count += 1
                            else:
                                self.null_responses += 1

                    except Exception:
                        self.failed_points += len(batch)

                # Progress guncelle
                if processed % 5 == 0:
                    self.points_remaining = (total_batches - processed) * self.BATCH_SIZE
                    self._emit_stats()

                # Kisa bekleme (CPU kullanimi icin)
                if not done_futures:
                    time.sleep(0.01)

        # Retry queue'yu isle (exponential backoff ile)
        if retry_queue and self.is_running:
            self.log.emit(f"  Retry: {len(retry_queue)} batch tekrar deneniyor...")

            # Retry'lari deneme sayisina gore grupla
            retry_groups = {}  # retry_count -> batches
            for batch, retry_count in retry_queue:
                retry_groups.setdefault(retry_count, []).append(batch)

            for retry_count in sorted(retry_groups.keys()):
                if not self.is_running:
                    break

                # Exponential backoff: 1s, 2s, 4s
                wait_time = 2 ** (retry_count - 1)
                self.log.emit(f"  Retry #{retry_count}: {wait_time}s beklenip {len(retry_groups[retry_count])} batch denenecek...")
                time.sleep(wait_time)

                for batch in retry_groups[retry_count]:
                    if not self.is_running:
                        break

                    time.sleep(0.3)  # Her retry arasinda bekle
                    results, meta, error = self._query_batch_single(batch)
                    self.api_calls += 1

                    if error:
                        self.failed_points += len(batch)
                        continue

                    # Hala hata varsa logla ama devam et
                    still_error = meta.get('error_coords', []) if meta else []
                    if still_error:
                        self.failed_points += len(still_error)
                        self.log.emit(f"    ⚠ {len(still_error)} koordinat hala hatali")

                    for i, result in enumerate(results):
                        if result and 'properties' in result:
                            if self._process_result(result, batch[i]):
                                new_count += 1
                        else:
                            self.null_responses += 1

        return new_count

    def _generate_grid(self, polygon: List[Tuple[float, float]], step_meters: int) -> List[Tuple[float, float]]:
        """Grid noktalari olustur."""
        bbox = KMLParser.get_bounding_box(polygon)
        center_lat = (bbox.min_lat + bbox.max_lat) / 2

        lat_step = step_meters / 111000
        lon_step = step_meters / (111000 * math.cos(math.radians(center_lat)))

        points = []
        lat = bbox.min_lat
        while lat <= bbox.max_lat:
            lon = bbox.min_lon
            while lon <= bbox.max_lon:
                if KMLParser.point_in_polygon(lat, lon, polygon):
                    points.append((lat, lon))
                lon += lon_step
            lat += lat_step

        return points

    def _generate_boundary_points(self, polygon: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """KML polygon siniri boyunca noktalar olustur."""
        points = []
        n = len(polygon)

        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]

            # Kenar uzunlugu
            edge_lat = (p1[0] + p2[0]) / 2
            lat_diff = (p2[0] - p1[0]) * 111000
            lon_diff = (p2[1] - p1[1]) * 111000 * math.cos(math.radians(edge_lat))
            edge_length = math.sqrt(lat_diff**2 + lon_diff**2)

            if edge_length < 1:
                continue

            # Kenar boyunca noktalar
            num_points = max(1, int(edge_length / self.boundary_step))

            for j in range(num_points + 1):
                t = j / max(num_points, 1)
                lat = p1[0] + t * (p2[0] - p1[0])
                lon = p1[1] + t * (p2[1] - p1[1])
                points.append((lat, lon))

        return points

    def _extract_parcel_edge_points(self, parcel_poly: List[Tuple[float, float]],
                                     search_polygon: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Parsel kenarlarindan disa noktalar cikar."""
        edge_points = []
        n = len(parcel_poly)
        outward_step = 8  # metre

        is_closed = (n >= 2 and parcel_poly[0] == parcel_poly[-1])
        edge_count = n - 1 if is_closed else n

        for i in range(edge_count):
            p1 = parcel_poly[i]
            p2 = parcel_poly[(i + 1) % n]

            edge_lat = (p1[0] + p2[0]) / 2
            lat_diff = (p2[0] - p1[0]) * 111000
            lon_diff = (p2[1] - p1[1]) * 111000 * math.cos(math.radians(edge_lat))
            edge_length = math.sqrt(lat_diff**2 + lon_diff**2)

            if edge_length < 1:
                continue

            num_points = max(1, int(edge_length / self.boundary_step))

            for j in range(num_points + 1):
                t = j / max(num_points, 1)
                mid_lat = p1[0] + t * (p2[0] - p1[0])
                mid_lon = p1[1] + t * (p2[1] - p1[1])

                # Normal vektor
                normal_x = lat_diff / edge_length
                normal_y = -lon_diff / edge_length

                for direction in [1, -1]:
                    delta_lat = direction * outward_step * normal_y / 111000
                    delta_lon = direction * outward_step * normal_x / (111000 * math.cos(math.radians(mid_lat)))

                    new_lat = mid_lat + delta_lat
                    new_lon = mid_lon + delta_lon

                    if (KMLParser.point_in_polygon(new_lat, new_lon, search_polygon) and
                            not KMLParser.point_in_polygon(new_lat, new_lon, parcel_poly)):
                        edge_points.append((new_lat, new_lon))

        return edge_points

    def _prune_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Bulunan parsellerin icindeki noktalari filtrele."""
        if not self.found_parcels:
            return points
        return [p for p in points
                if not any(KMLParser.point_in_polygon(p[0], p[1], poly) for poly in self.found_parcels)]

    def run(self):
        """Ana tarama algoritmasi."""
        quality_names = {
            ScanQuality.FAST: "Hizli",
            ScanQuality.BALANCED: "Dengeli",
            ScanQuality.DETAILED: "Detayli"
        }

        self.log.emit("=" * 50)
        self.log.emit("AKILLI TARAMA ALGORITMASI")
        self.log.emit("=" * 50)
        self.log.emit(f"Kalite: {quality_names[self.quality]}")
        self.log.emit(f"Kaba grid: {self.coarse_step}m, Yogun grid: {self.dense_step}m")
        self.log.emit(f"Paralel batch: {self.PARALLEL_BATCHES}")
        self.log.emit(f"Worker sayisi: {self.client.get_worker_count()}")

        total_found = 0

        for poly_idx, polygon in enumerate(self.polygons):
            if not self.is_running:
                break

            self.log.emit(f"\n{'='*50}")
            self.log.emit(f"Polygon {poly_idx + 1}/{len(self.polygons)}")
            self.log.emit("=" * 50)

            # === FAZ 1: KABA GRID ===
            self.current_phase = "Kaba Grid"
            self.log.emit(f"\n[FAZ 1] Kaba grid tarama ({self.coarse_step}m)...")

            coarse_points = self._generate_grid(polygon, self.coarse_step)
            self.log.emit(f"  {len(coarse_points)} nokta olusturuldu")

            phase1 = self._query_parallel(coarse_points)
            total_found += phase1
            self.log.emit(f"[FAZ 1] Tamamlandi: {phase1} parsel")
            self.progress.emit(30, 100)

            if not self.is_running:
                break

            # === FAZ 2: SINIR TARMA (KML + Parseller) ===
            self.current_phase = "Sinir"
            self.log.emit(f"\n[FAZ 2] Sinir tarama...")

            # KML polygon siniri
            boundary_points = self._generate_boundary_points(polygon)
            self.log.emit(f"  KML siniri: {len(boundary_points)} nokta")

            phase2a = self._query_parallel(boundary_points)
            total_found += phase2a

            # Bulunan parsellerin sinirlari (flood fill)
            if self.found_parcels:
                walked = set()
                parcels_to_walk = list(self.found_parcels)
                iteration = 0

                while parcels_to_walk and self.is_running and iteration < 5:
                    iteration += 1
                    current = parcels_to_walk
                    parcels_to_walk = []

                    all_edge_points = []
                    for parcel_poly in current:
                        if not parcel_poly or len(parcel_poly) < 3:
                            continue

                        poly_hash = tuple(sorted((round(p[0], 5), round(p[1], 5)) for p in parcel_poly))
                        if poly_hash in walked:
                            continue
                        walked.add(poly_hash)

                        edge_pts = self._extract_parcel_edge_points(parcel_poly, polygon)
                        all_edge_points.extend(edge_pts)

                    if not all_edge_points:
                        break

                    before = len(self.found_parcels)
                    new_found = self._query_parallel(all_edge_points)
                    total_found += new_found
                    after = len(self.found_parcels)

                    # Yeni parselleri kuyruga ekle
                    if after > before:
                        parcels_to_walk.extend(self.found_parcels[before:after])

                    self.log.emit(f"  Iterasyon {iteration}: {new_found} yeni parsel")

                    if new_found == 0:
                        break

            phase2 = total_found - phase1
            self.log.emit(f"[FAZ 2] Tamamlandi: {phase2} parsel")
            self.progress.emit(60, 100)

            if not self.is_running:
                break

            # === FAZ 3: YOGUN GRID (sadece bosluklarda) ===
            self.current_phase = "Yogun Grid"
            self.log.emit(f"\n[FAZ 3] Yogun grid ({self.dense_step}m)...")

            dense_points = self._generate_grid(polygon, self.dense_step)
            self.log.emit(f"  {len(dense_points)} nokta olusturuldu")

            # Pruning + duplicate filtre
            pruned = self._prune_points(dense_points)
            self.log.emit(f"  Pruning: {len(pruned)} nokta kaldi ({len(dense_points) - len(pruned)} elendi)")

            phase3 = self._query_parallel(pruned)
            total_found += phase3
            self.log.emit(f"[FAZ 3] Tamamlandi: {phase3} parsel")
            self.progress.emit(100, 100)

        # Ozet
        self.log.emit(f"\n{'='*50}")
        self.log.emit("TARAMA OZETI")
        self.log.emit("=" * 50)
        self.log.emit(f"Toplam parsel: {total_found}")
        self.log.emit(f"API cagrilari: {self.api_calls} batch ({self.total_points_queried} nokta)")
        self.log.emit(f"Tekrar sorgu engellendi: {self.duplicate_skipped}")
        self.log.emit(f"Bos yanit: {self.null_responses}")
        if self.failed_points > 0:
            self.log.emit(f"Hata: {self.failed_points} nokta")

        self.finished_scan.emit(total_found)

    def stop(self):
        """Taramayi durdur."""
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

        # Worker Durumu Card
        worker_card = CardWidget()
        worker_layout = QVBoxLayout(worker_card)
        worker_layout.addWidget(SubtitleLabel("Worker Durumu"))
        self.worker_status_label = BodyLabel("workers.json yükleniyor...")
        worker_layout.addWidget(self.worker_status_label)
        layout.addWidget(worker_card)

        # Worker durumunu kontrol et
        self._check_worker_config()

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

        # Tarama Kalitesi Card
        quality_card = CardWidget()
        quality_layout = QVBoxLayout(quality_card)
        quality_layout.addWidget(SubtitleLabel("Tarama Kalitesi"))

        # Radio butonlar
        radio_layout = QHBoxLayout()

        self.quality_group = QButtonGroup(self)

        self.radio_fast = RadioButton("Hizli")
        self.radio_fast.setToolTip("Buyuk alanlar icin (100m → 40m grid)")
        self.quality_group.addButton(self.radio_fast, 0)
        radio_layout.addWidget(self.radio_fast)

        self.radio_balanced = RadioButton("Dengeli (Onerilen)")
        self.radio_balanced.setToolTip("Cogu durum icin ideal (60m → 20m grid)")
        self.radio_balanced.setChecked(True)
        self.quality_group.addButton(self.radio_balanced, 1)
        radio_layout.addWidget(self.radio_balanced)

        self.radio_detailed = RadioButton("Detayli")
        self.radio_detailed.setToolTip("Kucuk parseller icin (40m → 10m grid)")
        self.quality_group.addButton(self.radio_detailed, 2)
        radio_layout.addWidget(self.radio_detailed)

        radio_layout.addStretch()
        quality_layout.addLayout(radio_layout)

        layout.addWidget(quality_card)

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

    def _check_worker_config(self):
        """workers.json dosyasını kontrol eder ve durumu gösterir."""
        try:
            client = TKGMClient.from_config()
            n = client.get_worker_count()
            self.worker_status_label.setText(
                f"{n} worker aktif (paralel: {client.parallel_tries})"
            )
            self._worker_config_ok = True
        except FileNotFoundError:
            self.worker_status_label.setText(
                "workers.json bulunamadi! Lutfen olusturun."
            )
            self._worker_config_ok = False
        except Exception as e:
            self.worker_status_label.setText(f"Hata: {str(e)[:50]}")
            self._worker_config_ok = False

    def load_settings(self):
        """Kaydedilmis ayarlari yukler."""
        # Kalite ayari
        quality = self.settings.value("quality", 1, type=int)
        if quality == 0:
            self.radio_fast.setChecked(True)
        elif quality == 2:
            self.radio_detailed.setChecked(True)
        else:
            self.radio_balanced.setChecked(True)

    def save_settings(self):
        """Mevcut ayarlari kaydeder."""
        self.settings.setValue("quality", self.quality_group.checkedId())

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

    def _get_selected_quality(self) -> ScanQuality:
        """Secili kalite seviyesini dondurur."""
        quality_id = self.quality_group.checkedId()
        if quality_id == 0:
            return ScanQuality.FAST
        elif quality_id == 2:
            return ScanQuality.DETAILED
        return ScanQuality.BALANCED

    def load_kml(self, path: str):
        """KML dosyasini yukler ve parse eder."""
        try:
            self.polygons = KMLParser.parse(path)
            info = f"{len(self.polygons)} polygon bulundu"

            # Tahmini alan hesapla (dengeli kalite icin)
            total_area_km2 = 0
            for poly in self.polygons:
                bbox = KMLParser.get_bounding_box(poly)
                center_lat = (bbox.min_lat + bbox.max_lat) / 2
                width_km = (bbox.max_lon - bbox.min_lon) * 111 * math.cos(math.radians(center_lat))
                height_km = (bbox.max_lat - bbox.min_lat) * 111
                total_area_km2 += width_km * height_km

            info += f" (~{total_area_km2:.2f} km²)"
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

        # Worker config kontrolü
        if not getattr(self, '_worker_config_ok', False):
            InfoBar.error(
                title="Hata",
                content="workers.json bulunamadi veya hatali!",
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000
            )
            return

        self.save_settings()
        self.parcels = {}  # Yeni tarama, onceki parselleri temizle

        self._start_worker(reuse_parcels=False)
        self.log("Tarama baslatildi...")

    def _start_worker(self, reuse_parcels: bool = False, quality: ScanQuality = None):
        """Worker olusturur ve baslatir."""
        client = TKGMClient.from_config()

        # Kalite secimi
        if quality is None:
            quality = self._get_selected_quality()

        self.worker = ScanWorker(client, self.polygons, quality)

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
        skipped = stats.get("skipped", 0)
        null_resp = stats.get("null_responses", 0)
        failed = stats.get("failed_points", 0)

        parts = [f"Faz: {phase}", f"Sorgulanan: {queried}"]
        if remaining > 0:
            parts.append(f"Kalan: {remaining}")
        if skipped > 0:
            parts.append(f"Atlanan: {skipped}")
        parts.append(f"Bos: {null_resp}")
        if failed > 0:
            parts.append(f"Hata: {failed}")

        self.stats_label.setText(" | ".join(parts))

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
        """Eksik bolgeleri Detayli kalite ile tekrar tarar."""
        if not self.polygons:
            InfoBar.warning(
                title="Uyari",
                content="Taranacak polygon yok",
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000
            )
            return

        if not self._worker_config_ok:
            InfoBar.error(
                title="Hata",
                content="workers.json yapilandirmasi hatali",
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000
            )
            return

        self.log("\n=== DETAYLANDIRMA TARAMASI ===")
        self.log(f"Detayli kalite ile yeniden taranacak ({len(self.parcels)} mevcut parsel korunuyor)")

        self._start_worker(reuse_parcels=True, quality=ScanQuality.DETAILED)

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
