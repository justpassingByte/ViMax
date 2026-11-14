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

## Phase 01 — Bám sát cốt truyện (cờ `quality.canon_gate`)
- **Thay đổi:**
  - `agents/storyboard_artist.py`: thêm template `system_prompt_template_design_storyboard_grounded`
    (giữ định dạng/thoại/JSON, bỏ các khối ép bịa: 5 bước đánh nhau, B-roll, 4-Phase Sakuga,
    Rescue Exclusivity, Sky Streaks, Outerwear Value, BAN EYE-LEVEL; thêm 6 luật căn cứ).
    `design_storyboard(grounded=)` chọn template; `review_and_repair_storyboard(grounded=)` thêm luật 8.
  - `agents/canon_gate.py` (mới, thuần code): `check_storyboard` phát hiện 8 loại lỗi
    (uncited, unknown_beat, invented/missing/duplicate_dialogue, foreign_character, order_violation,
    coverage_ratio); `repair_storyboard` sửa cố định (bỏ shot lỗi, chèn shot cho thoại thiếu, sắp lại theo beat).
  - `pipelines/script2video_pipeline.py::design_storyboard`: thêm `grounded=` và `force=` (bỏ qua cache khi thử lại).
  - `pipelines/manga2video_pipeline.py::plan_episode`: `_apply_canon_gate` chạy sau review, trước bind
    (thử lại tối đa 2 lần rồi repair), ghi `scene_N/canon_report.json`.
  - `agents/cinematic_director.py::plan_scene_expansion(restrict_shot_types=)`: giới hạn shot phụ
    (chỉ establishing_wide ở beat đầu/scene-break, reaction/insert/OTS/alternate_angle; ≤ 25% số beat).
  - `interfaces/manga.py`: thêm `MangaDialogueLine.source_text`; `agents/manga_reader.py`: thêm yêu cầu điền.
  - `tests/test_canon_gate.py`: 6 test (đủ 8 loại lỗi, flashback, repair→passed, chọn template theo cờ).
- **Tắt / lùi:** đặt `quality.canon_gate: false` → dùng template cũ, không chạy cổng cốt truyện,
  không giới hạn cinematic director. Cache cũ không có `source_text` vẫn chạy (dùng `line`+`vietsub`).
- **Test:** `pytest tests -q` → 363 passed.
