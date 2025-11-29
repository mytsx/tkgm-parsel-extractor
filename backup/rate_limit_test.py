#!/usr/bin/env python3
"""
TKGM API Rate Limit Test
Lokal makineden direkt TKGM API'ye istek atarak rate limit davranışını test eder.
"""

import requests
import time
from datetime import datetime

# Farklı koordinatlar üret (her istek benzersiz olsun)
def generate_unique_coords(count: int):
    """Türkiye genelinde benzersiz koordinatlar üret."""
    import random
    coords = []
    # Türkiye sınırları: lat 36-42, lon 26-45
    for i in range(count):
        # Her seferinde farklı koordinat
        lat = 36.0 + (i * 0.01) % 6  # 36-42 arası
        lon = 26.0 + (i * 0.013) % 19  # 26-45 arası
        # Küçük random ekleme (tamamen benzersiz olsun)
        lat += random.uniform(0.0001, 0.0009)
        lon += random.uniform(0.0001, 0.0009)
        coords.append((round(lat, 6), round(lon, 6)))
    return coords

TEST_COORDS = generate_unique_coords(500)  # 500 benzersiz koordinat

# Direkt TKGM API veya Worker üzerinden test
TKGM_API_DIRECT = "https://cbsapi.tkgm.gov.tr/megsiswebapi.v3.1/api/parsel"
WORKER_URL = None  # Worker URL'si (varsa)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://parselsorgu.tkgm.gov.tr/",
    "Origin": "https://parselsorgu.tkgm.gov.tr"
}


def test_single_request(lat: float, lon: float, request_num: int, worker_url: str = None, api_key: str = None) -> dict:
    """Tek istek at ve sonucu döndür."""
    if worker_url:
        url = f"{worker_url}/parsel/{lat}/{lon}"
    else:
        url = f"{TKGM_API_DIRECT}/{lat}/{lon}/"
    start = time.time()

    headers = HEADERS.copy()
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        response = requests.get(url, headers=headers, timeout=10)
        elapsed = (time.time() - start) * 1000  # ms

        return {
            "num": request_num,
            "status": response.status_code,
            "elapsed_ms": round(elapsed),
            "has_data": bool(response.text and len(response.text) > 10),
            "size": len(response.text),
            "error": None
        }
    except requests.exceptions.Timeout:
        return {"num": request_num, "status": "TIMEOUT", "error": "Timeout"}
    except requests.exceptions.ConnectionError as e:
        return {"num": request_num, "status": "CONN_ERR", "error": str(e)[:50]}
    except Exception as e:
        return {"num": request_num, "status": "ERROR", "error": str(e)[:50]}


def run_test(delay_ms: int = 0, max_requests: int = 100, worker_url: str = None, api_key: str = None):
    """Rate limit testi çalıştır."""
    print("=" * 60)
    print("TKGM API RATE LIMIT TESTİ")
    print(f"Hedef: {'Worker: ' + worker_url if worker_url else 'Direkt TKGM API'}")
    if api_key:
        print(f"API Key: {api_key[:20]}...")
    print(f"Delay: {delay_ms}ms | Max: {max_requests} istek")
    print(f"Başlangıç: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    results = {
        "success": 0,      # 200 + veri var
        "empty": 0,        # 200 + veri yok
        "rate_limited": 0, # 429 veya benzeri
        "error": 0,        # Diğer hatalar
        "timeout": 0
    }

    first_error_at = None

    for i, (lat, lon) in enumerate(TEST_COORDS[:max_requests]):
        result = test_single_request(lat, lon, i + 1, worker_url, api_key)

        # Sonucu kategorize et
        status = result["status"]
        if status == 200:
            if result["has_data"]:
                results["success"] += 1
                status_char = "✓"
            else:
                results["empty"] += 1
                status_char = "○"
        elif status == 429:
            results["rate_limited"] += 1
            status_char = "⛔"
            if not first_error_at:
                first_error_at = i + 1
        elif status in ["TIMEOUT", "CONN_ERR", "ERROR"]:
            if status == "TIMEOUT":
                results["timeout"] += 1
            else:
                results["error"] += 1
            status_char = "✗"
            if not first_error_at:
                first_error_at = i + 1
        else:
            results["error"] += 1
            status_char = "?"
            if not first_error_at:
                first_error_at = i + 1

        # Her istek için kısa log
        elapsed = result.get("elapsed_ms", "?")
        print(f"  [{i+1:3d}] {status_char} {status} - {elapsed}ms", end="")
        if result.get("error"):
            print(f" - {result['error']}", end="")
        print()

        # Her 10 istekte özet
        if (i + 1) % 10 == 0:
            print(f"  --- {i+1} istek: {results['success']} başarılı, {results['empty']} boş, {results['rate_limited']} engel, {results['error']+results['timeout']} hata ---")

        # Rate limit yedik mi? 5 ardışık hata varsa dur
        if results["rate_limited"] >= 5 or results["error"] + results["timeout"] >= 5:
            recent_errors = sum(1 for r in [result] if r["status"] != 200)
            if recent_errors >= 3:
                print(f"\n⚠️  Çok fazla hata, test durduruluyor...")
                break

        # Delay uygula
        if delay_ms > 0:
            time.sleep(delay_ms / 1000)

    # Sonuç özeti
    print("\n" + "=" * 60)
    print("SONUÇ ÖZETİ")
    print("=" * 60)
    total = sum(results.values())
    print(f"Toplam istek:     {total}")
    print(f"Başarılı (veri):  {results['success']}")
    print(f"Başarılı (boş):   {results['empty']}")
    print(f"Rate Limited:     {results['rate_limited']}")
    print(f"Timeout:          {results['timeout']}")
    print(f"Diğer Hata:       {results['error']}")
    if first_error_at:
        print(f"\nİlk hata: {first_error_at}. istekte")
    print(f"Bitiş: {datetime.now().strftime('%H:%M:%S')}")

    return results


def run_multi_worker_test(workers: list, api_key: str, delay_ms: int = 0, max_requests: int = 200):
    """Birden fazla worker ile round-robin test."""
    print("=" * 60)
    print(f"MULTI-WORKER RATE LIMIT TESTİ ({len(workers)} worker)")
    print("=" * 60)
    for i, w in enumerate(workers):
        print(f"  Worker {i+1}: {w}")
    print(f"Delay: {delay_ms}ms | Max: {max_requests} istek")
    print(f"Başlangıç: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    results = {"success": 0, "empty": 0, "rate_limited": 0, "error": 0, "timeout": 0}
    worker_stats = {w: {"success": 0, "error": 0} for w in workers}

    for i, (lat, lon) in enumerate(TEST_COORDS[:max_requests]):
        # Round-robin worker seçimi
        worker = workers[i % len(workers)]
        result = test_single_request(lat, lon, i + 1, worker, api_key)

        status = result["status"]
        if status == 200:
            results["success"] += 1
            worker_stats[worker]["success"] += 1
            status_char = "✓"
        elif status in [403, 429]:
            results["error"] += 1
            worker_stats[worker]["error"] += 1
            status_char = "⛔"
        else:
            results["error"] += 1
            worker_stats[worker]["error"] += 1
            status_char = "?"

        w_idx = workers.index(worker) + 1
        print(f"  [{i+1:3d}] W{w_idx} {status_char} {status} - {result.get('elapsed_ms', '?')}ms")

        if (i + 1) % 20 == 0:
            print(f"  --- {i+1} istek: {results['success']} başarılı, {results['error']} hata ---")

        if delay_ms > 0:
            time.sleep(delay_ms / 1000)

    print("\n" + "=" * 60)
    print("SONUÇ ÖZETİ")
    print("=" * 60)
    print(f"Toplam: {sum(results.values())} | Başarılı: {results['success']} | Hata: {results['error']}")
    print("\nWorker bazlı:")
    for i, (w, stats) in enumerate(worker_stats.items()):
        total = stats['success'] + stats['error']
        rate = (stats['success'] / total * 100) if total > 0 else 0
        print(f"  W{i+1}: {stats['success']}/{total} başarılı ({rate:.1f}%)")

    return results


if __name__ == "__main__":
    import sys
    from PyQt5.QtCore import QSettings

    # Ayarları oku
    settings = QSettings('TKGM', 'ParselCekme')
    saved_worker = settings.value('worker_url', '')
    saved_api_key = settings.value('api_key', '')

    # Kullanım:
    #   python rate_limit_test.py [delay_ms] [max_requests] [mode]
    #   mode: "direct" | "worker" | "multi"
    delay = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    max_req = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    mode = sys.argv[3] if len(sys.argv) > 3 else "worker"

    if mode == "direct":
        print("\n🔬 DİREKT TKGM API TESTİ")
        run_test(delay_ms=delay, max_requests=max_req, worker_url=None, api_key=None)

    elif mode == "multi" or mode.startswith("multi"):
        # Multi-worker test
        # mode: "multi" = 3 worker, "multi8" = 8 worker, vs.
        num_workers = 8 if "8" in mode else (int(mode[5:]) if len(mode) > 5 and mode[5:].isdigit() else 3)

        base_url = saved_worker.rstrip('/')
        if 'tkgm-proxy' in base_url:
            workers = [
                base_url.replace('tkgm-proxy', f'tkgm-proxy-{i}')
                for i in range(1, num_workers + 1)
            ]
        else:
            workers = [base_url]

        print(f"\n🔬 MULTI-WORKER TESTİ ({num_workers} worker)")
        run_multi_worker_test(workers, saved_api_key, delay_ms=delay, max_requests=max_req)

    else:
        print(f"\n🔬 WORKER TESTİ: {saved_worker}")
        run_test(delay_ms=delay, max_requests=max_req, worker_url=saved_worker, api_key=saved_api_key)
