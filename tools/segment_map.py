#!/usr/bin/env python3
"""Phan vung ban do Occupancy Grid (.pgm + map.yaml) bang distance transform + watershed.

Dau ra:
  zones.yaml        - danh sach khu vuc (id, ten, alias, toa do dich hop le trong frame 'map')
  zones_labeled.png - anh ban do to mau + nhan khu vuc

Hai che do:
  * Tu dong : seed = vung cach tuong > --seed-dist (m). Cua hep hon 2*seed-dist se tach phong.
  * Thu cong: --seeds seeds.yaml, moi seed la mot diem (x, y) trong toa do map + ten khu vuc.
"""
import argparse
import math
import os

import cv2
import numpy as np
import yaml

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_map(yaml_path):
    with open(yaml_path, encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    img_path = meta["image"]
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)
    return meta, img


def free_mask(img, meta):
    """Chi coi o 'free' that su (254); o unknown (205) va o co vat can bi loai."""
    if int(meta.get("negate", 0)):
        return (img <= 5).astype(np.uint8)
    return (img >= 250).astype(np.uint8)


def px_to_world(col, row, meta, h):
    res = meta["resolution"]
    ox, oy = meta["origin"][:2]
    return float(ox + (col + 0.5) * res), float(oy + (h - row - 0.5) * res)


def world_to_px(x, y, meta, h):
    res = meta["resolution"]
    ox, oy = meta["origin"][:2]
    col = int(math.floor((x - ox) / res))
    row = h - 1 - int(math.floor((y - oy) / res))
    return col, row


def segment(free, dist, res, seed_dist_m, seeds_px=None):
    """Tra ve anh nhan int32: 0 = tuong/khong xac dinh, >=2 = khu vuc."""
    markers = np.zeros(free.shape, np.int32)
    if seeds_px:
        for i, (c, r) in enumerate(seeds_px):
            cv2.circle(markers, (c, r), max(2, int(0.15 / res)), i + 2, -1)
    else:
        sure = (dist > seed_dist_m / res).astype(np.uint8)
        n, cc = cv2.connectedComponents(sure)
        markers = np.where(cc > 0, cc + 1, 0).astype(np.int32)
    if seeds_px:  # co seed (thu cong / k-means): chia theo khoang cach duong di ngan nhat
        return flood_geodesic(markers, free, dist, res)
    return flood(markers, free, dist)


def flood_geodesic(markers, free, dist, res):
    """Dijkstra da nguon tu cac seed. Buoc di sat tuong (<0.5 m) ton chi phi cao hon, nen cho hep
    (cua) giong 'rao can' va ranh gioi giua cac vung nghieng ve phia do."""
    import heapq
    h, w = free.shape
    cost = 1.0 + 4.0 * np.clip(1.0 - dist / (0.5 / res), 0.0, 1.0)
    D = np.full((h, w), np.inf)
    lab = np.zeros((h, w), np.int32)
    heap = []
    for r, c in zip(*np.nonzero((markers >= 2) & (free > 0))):
        lab[r, c] = markers[r, c]
        D[r, c] = 0.0
        heap.append((0.0, int(r), int(c)))
    heapq.heapify(heap)
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > D[r, c]:
            continue
        l = lab[r, c]
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < h and 0 <= cc < w and free[rr, cc]:
                nd = d + cost[rr, cc]
                if nd < D[rr, cc]:
                    D[rr, cc] = nd
                    lab[rr, cc] = l
                    heapq.heappush(heap, (nd, rr, cc))
    return lab


def flood(markers, free, dist):
    """Watershed tren -dist: lan nhan tu seed, pixel xa tuong nhat duoc xu ly truoc,
    nen ranh gioi giua 2 vung nam o cho hep nhat (cua)."""
    import heapq
    h, w = free.shape
    lab = np.zeros((h, w), np.int32)
    heap = []
    for r, c in zip(*np.nonzero((markers >= 2) & (free > 0))):
        lab[r, c] = markers[r, c]
        heap.append((-float(dist[r, c]), int(r), int(c)))
    heapq.heapify(heap)
    while heap:
        _, r, c = heapq.heappop(heap)
        l = lab[r, c]
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):  # 4-lang gieng
            if 0 <= rr < h and 0 <= cc < w and free[rr, cc] and lab[rr, cc] == 0:
                lab[rr, cc] = l
                heapq.heappush(heap, (-float(dist[rr, cc]), rr, cc))
    return lab


def kmeans_seeds(free, dist, res, k):
    """Cho khong gian mo (khong co cua/phong): k-means tren toa do o free, roi dua moi tam
    ve o free gan nhat va cach tuong >= 0.4 m de lam seed cho watershed."""
    ys, xs = np.nonzero(free)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    if len(pts) > 30000:
        pts = pts[np.random.RandomState(0).choice(len(pts), 30000, replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.5)
    cv2.setRNGSeed(0)
    _, _, centers = cv2.kmeans(pts, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    ok_r, ok_c = np.nonzero(dist >= 0.4 / res)
    ok = np.column_stack([ok_c, ok_r]).astype(np.float32)
    seeds = []
    for c in centers:
        j = np.argmin(((ok - c) ** 2).sum(1))
        seeds.append((int(ok[j][0]), int(ok[j][1])))
    return seeds


def default_aliases(zid):
    z = zid.lower()
    return [f"khu vực {z}", f"khu {z}", f"vùng {z}", f"zone {z}"]


def extract_zones(labels, dist, meta, res, min_area_m2, seeds_meta=None):
    h = labels.shape[0]
    zones = []
    for lab in sorted(set(np.unique(labels)) - {0}):
        mask = labels == lab
        area = mask.sum() * res * res
        if area < min_area_m2:
            continue
        ys, xs = np.nonzero(mask)
        cx, cy = px_to_world(xs.mean(), ys.mean(), meta, h)
        # Diem dich hop le = diem trong vung xa vat can nhat (chac chan nam o o free)
        d = np.where(mask, dist, 0)
        r, c = np.unravel_index(np.argmax(d), d.shape)
        gx, gy = px_to_world(c, r, meta, h)
        zones.append({
            "label": int(lab), "cx": cx, "cy": cy, "gx": gx, "gy": gy,
            "area": float(area), "clearance": float(d[r, c] * res),
            "px": (int(c), int(r)),
        })

    if seeds_meta:
        zones.sort(key=lambda z: z["label"])
        for z in zones:
            sm = seeds_meta[z["label"] - 2]
            z["id"] = str(sm.get("id", LETTERS[z["label"] - 2]))
            z["name"] = sm.get("name", f"Khu vực {z['id']}")
            z["aliases"] = list(sm.get("aliases", [])) + default_aliases(z["id"])
    else:
        zones.sort(key=lambda z: (z["cx"], z["cy"]))
        for i, z in enumerate(zones):
            z["id"] = LETTERS[i] if i < len(LETTERS) else f"Z{i}"
            z["name"] = f"Khu vực {z['id']}"
            z["aliases"] = default_aliases(z["id"])
    return zones


def draw(img, labels, zones, scale):
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay = color.copy()
    for z in zones:
        hue = int((z["label"] * 47) % 180)
        bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        overlay[labels == z["label"]] = bgr
    out = cv2.addWeighted(overlay, 0.55, color, 0.45, 0)
    out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for z in zones:
        c, r = z["px"][0] * scale, z["px"][1] * scale
        cv2.circle(out, (c, r), 5, (0, 0, 255), -1)
        cv2.putText(out, z["id"], (c + 8, r - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, z["id"], (c + 8, r - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--map", required=True, help="duong dan map.yaml")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--seed-dist", type=float, default=0.6,
                    help="(tu dong) khoang cach toi thieu toi tuong de lam seed, don vi m")
    ap.add_argument("--min-area", type=float, default=1.0, help="dien tich vung toi thieu (m2)")
    ap.add_argument("--seeds", help="(thu cong) file seeds.yaml")
    ap.add_argument("--k", type=int, default=0,
                    help="(tu dong cho khong gian mo) chia thanh k vung bang k-means + watershed")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    meta, img = load_map(a.map)
    h, w = img.shape
    res = float(meta["resolution"])

    free = free_mask(img, meta)
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
    for i in range(1, n):  # bo cac dom free nho (nhieu laser)
        if stats[i, cv2.CC_STAT_AREA] * res * res < a.min_area:
            free[cc == i] = 0
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)

    seeds_px, seeds_meta = None, None
    if a.seeds:
        with open(a.seeds, encoding="utf-8") as f:
            seeds_meta = yaml.safe_load(f)["seeds"]
        seeds_px = [world_to_px(s["x"], s["y"], meta, h) for s in seeds_meta]

    if a.k and not seeds_px:
        seeds_px = kmeans_seeds(free, dist, res, a.k)
    labels = segment(free, dist, res, a.seed_dist, seeds_px)
    zones = extract_zones(labels, dist, meta, res, a.min_area, seeds_meta)
    if not zones:
        raise SystemExit("Khong tim thay khu vuc nao - thu giam --seed-dist hoac --min-area")

    os.makedirs(a.out_dir, exist_ok=True)
    out = {"map_frame": "map", "zones": []}
    for z in zones:
        out["zones"].append({
            "id": z["id"], "name": z["name"], "aliases": z["aliases"],
            "goal": {"x": round(z["gx"], 3), "y": round(z["gy"], 3), "yaw": 0.0},
            "centroid": {"x": round(z["cx"], 3), "y": round(z["cy"], 3)},
            "area_m2": round(z["area"], 2), "clearance_m": round(z["clearance"], 2),
        })
        flag = "  (CANH BAO: sat tuong, nen dat lai seed)" if z["clearance"] < 0.3 else ""
        print(f"[{z['id']}] {z['name']}: goal=({z['gx']:.2f}, {z['gy']:.2f}) "
              f"area={z['area']:.1f} m2 clearance={z['clearance']:.2f} m{flag}")
    with open(os.path.join(a.out_dir, "zones.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)

    scale = max(1, 900 // max(h, w))
    vis = draw(img, labels, zones, scale)
    cv2.imwrite(os.path.join(a.out_dir, "zones_labeled.png"), vis)
    print(f"Da ghi {a.out_dir}/zones.yaml va zones_labeled.png")
    if a.show:
        cv2.imshow("zones", vis)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
