# Incremental Quality — Nhật ký thay đổi

Mỗi phase một đoạn: đã đổi gì, file nào, cờ nào để tắt nếu thấy kém hơn.
Mặc định mọi cờ `quality.*` = bản nâng cấp. Để lùi một phần, đổi cờ tương ứng
trong `configs/agent.local.yaml` (hoặc đặt biến môi trường `VIMAX_QUALITY_*`).

## Baseline
- Commit `chore: snapshot manga_vision foundation WIP as baseline` gói lại toàn bộ
  công việc đang dở (pipeline manga_vision, guard, xử lý payload ảnh/video muse2api,
  phụ đề có gắn tên người nói) để các phase sau đứng trên nền xanh.
- Cập nhật 5 test đã lệch so với hành vi mới (JPEG data-URL, attribution phụ đề,
  `shot_requires_last_frame` bỏ last-frame cho shot nhỏ/sakuga).

## Phase 00 — Chuẩn bị
- **Thay đổi:** thêm section cờ `quality:` và accessor đọc cấu hình.
  - `agent_runtime/config.py`: 20 hàm `quality_*()` theo đúng kiểu `manga_vision_*`
    (ưu tiên biến môi trường `VIMAX_QUALITY_*` → section `quality:` → mặc định nâng cấp).
  - `configs/agent.example.yaml`: section `quality:` kèm chú thích từng cờ.
  - `configs/agent.local.yaml`: thêm `quality.series_slug: eleceed` (các cờ khác dùng mặc định).
  - `tests/test_agent_config.py`: `QualityConfigTests` (mặc định / override config / enum sai / env override).
- **Giữ output cũ:** không đụng `.working_dir/`. Lần chạy nâng cấp dùng working dir mới.
- **Tắt / lùi:** chưa có hành vi pipeline nào đổi ở phase này; chỉ thêm cờ.
- **Test:** `pytest tests -q` → 357 passed.
