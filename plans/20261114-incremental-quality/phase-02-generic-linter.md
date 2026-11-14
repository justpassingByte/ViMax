# Phase 02 — Thay guard bằng bộ kiểm tra dùng chung, gỡ code riêng Eleceed

Cờ: `quality.generic_linter`, thêm `quality.series_slug` (ví dụ `eleceed`)

## Vì sao
`agents/anime_director_guard.py` (1.302 dòng) trộn hai loại việc:
- **Việc chung, nên giữ:** xóa bong bóng/chữ/khung truyện khỏi prompt (`sanitize_comic_panel_artifacts`), làm an toàn prompt chuyển động (`sanitize_motion_prompt_safety`), phát hiện nhảy bối cảnh (`detect_context_jump`), một phần `audit_storyboard_defects` (nhân bản nhân vật, panel chia đôi, nền trắng trống, 4+ shot cùng `cam_idx`).
- **Việc riêng Eleceed hoặc có hại:** `purify_and_enforce_dynamic_camera_angles` (đổi mọi medium/eye-level thành góc thấp → góc máy lặp lại), `enforce_jiwoo_wardrobe`, `enforce_hojin_and_mother_wardrobe`, `enforce_pet_shop_and_store_owner`, `enforce_kayden_cat_bandages`, `enforce_character_injury_continuity`, `enforce_character_facial_integrity`, `enforce_gaze_direction_continuity`, `enforce_anime_style` (bảng màu Eleceed).

Code riêng Eleceed còn nằm ngoài guard:
- `pipelines/script2video_pipeline.py`: ~dòng 1434–1452 và ~1945–1960 gọi các hàm `enforce_*` lên prompt keyframe; `_ensure_character_portraits_and_spatial_anchors` (~dòng 497) có nhánh Kayden; ~dòng 1395 mô tả "slate-grey coat / alley"; ~dòng 320–348 `_resolve_character_acoustic_profile` viết cứng giọng nhân vật; ~dòng 1600 kiểm tra "kayden (human)" trong prompt video.
- `utils/audio.py` ~dòng 194–262: map giọng Jiwoo/Kayden.
- `agents/reference_image_selector.py`: prompt nhắc "Eleceed" và "Kayden in prologue scenes 0-4".
- `agents/manga_reader.py` import guard chỉ để gọi `detect_context_jump`.

## Thay đổi

### 2.1 Module mới `agents/shot_linter.py`
Chuyển (copy rồi làm sạch) phần việc chung: `strip_comic_artifacts`, `sanitize_motion_prompt`, `detect_context_jump`, `audit_storyboard` (chỉ các kiểm tra chung ở trên). **Không** đổi góc máy, **không** chèn mô tả trang phục/nhân vật. Không có tên nhân vật nào trong file.

### 2.2 Dữ liệu bộ truyện `series/<slug>/`
```
series/eleceed/series.yaml       # title, style_prompt (bảng màu chuyển từ enforce_anime_style), negative_prompt
series/eleceed/characters.yaml   # mỗi nhân vật: id, aliases, prompt_notes, voice_description, wardrobe_rules
```
- `prompt_notes`: chuyển nội dung từ `enforce_character_facial_integrity` (Kayden không râu…), `enforce_kayden_cat_bandages`, phần mô tả Kayden trong `_ensure_character_portraits_and_spatial_anchors`.
- `wardrobe_rules`: danh sách `{from_scene, to_scene, outfit_text}` chuyển từ `enforce_jiwoo_wardrobe`, `enforce_hojin_and_mother_wardrobe`, `enforce_pet_shop_and_store_owner`. (Phase 05 thay bằng trạng thái theo beat.)
- `voice_description`: chuyển từ `_resolve_character_acoustic_profile` và `utils/audio.py`.
- Loader `utils/series_bible.py`: `load_series(slug)`, `style_prompt()`, `character_notes(name_or_alias, scene_idx)`, `voice_description(name_or_alias)`. Không có bộ truyện → trả rỗng, pipeline vẫn chạy.

### 2.3 Đấu nối khi `generic_linter: true`
- `pipelines/manga2video_pipeline.py`: thay các lời gọi `AnimeDirectorGuard.*` (~dòng 764, 768, 791, 828, 847–852) bằng `shot_linter.*`.
- `agents/storyboard_artist.py::review_and_repair_storyboard`: dùng `shot_linter.audit_storyboard`.
- `pipelines/script2video_pipeline.py`: thay chuỗi `enforce_*` bằng: `prompt += series_bible.style_prompt()` + `character_notes(...)` cho từng nhân vật có mặt. Bỏ nhánh Kayden trong `_ensure_character_portraits_and_spatial_anchors` và trong prompt video; mô tả anchor dùng câu chung. `_resolve_character_acoustic_profile` đọc `voice_description` từ series bible, giữ câu mặc định chung.
- `utils/audio.py`: map giọng lấy từ `audio.character_voices` (đã có) hoặc series bible; bỏ nhánh tên cứng.
- `agents/reference_image_selector.py`: bỏ câu "Eleceed"/"Kayden" khỏi prompt; phong cách lấy từ `style_prompt()`.
- `agents/manga_reader.py`: import `detect_context_jump` từ `shot_linter`.
- Mặc định `true`. Khi người dùng tắt cờ: chạy đường cũ (để quay lại nếu thấy kém hơn). `anime_director_guard.py` chỉ xóa ở phase 10.

### 2.4 Test chặn tên bộ truyện trong code
`tests/test_no_series_names_in_code.py`: đọc mọi tên và alias trong `series/*/characters.yaml` + `title` trong `series.yaml`; quét `agents/ pipelines/ utils/ tools/ interfaces/ agent_runtime/`; fail nếu thấy (không phân biệt hoa thường, theo ranh giới từ). Cho phép danh sách ngoại lệ tạm thời `tests/series_name_allowlist.txt` chỉ chứa file `anime_director_guard.py` cho tới phase 10.

## Test
- Test `shot_linter` với các chuỗi lấy từ `tests/test_anime_director_guard.py` thuộc nhóm "việc chung" → kết quả tương đương.
- Test `series_bible` với thư mục fixture.
- Test khi không có `series_slug` → prompt không có ghi chú nhân vật, không lỗi.
