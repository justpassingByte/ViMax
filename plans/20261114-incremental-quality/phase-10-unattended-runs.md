# Phase 10 — Chạy cả chương / nhiều chương không cần người

Cờ: `quality.exceptions_queue`; thêm `quality.max_clip_retakes` (mặc định 1), `quality.chapter_budget_images`, `quality.chapter_budget_clips`

## Mục tiêu
Một lệnh chạy hết nhiều chương; lỗi không làm dừng cả lượt chạy; người dùng chỉ xem một trang ngoại lệ mỗi chương, ≤ 5 phút.

## Thay đổi

### 10.1 QA clip
- `agents/identity_qa.py::check_clip(video_path, shot, ...)`: lấy frame đầu, giữa, cuối (moviepy/ffmpeg đã có) → `check_frame` cho từng frame.
- Nếu `voice_mode != native` và có nhóm `voice_eval`: đo độ giống giọng của đoạn lời so với hồ sơ.
- Rớt: tạo lại clip (tối đa `max_clip_retakes`); vẫn rớt → dự phòng bằng keyframe + chuyển động Ken Burns có sẵn (`utils/anime_fx.py::apply_anime_ken_burns_drift`), ghi ngoại lệ.

### 10.2 Hàng ngoại lệ
- Gom `scene_N/exceptions.json` (phase 06), `profile_merge_suggestions.json` (phase 05), `story_memory/conflicts.json` (phase 09), `canon_report.json` có repair (phase 01) thành `exceptions/chNN.json` của chương.
- `tools/review_page.py`: sinh HTML một trang cho chương: mỗi ngoại lệ có ảnh/clip, lý do, và lựa chọn (chấp nhận / làm lại shot / ghép hồ sơ / bỏ qua). Lưu quyết định ra `exceptions/chNN.decisions.json`.
- `python -m pipelines.apply_decisions <series_dir> chNN`: áp quyết định (xóa frame để làm lại, gọi `profile_store.merge`, đánh dấu đã xem). Chạy lại chương chỉ làm lại phần bị ảnh hưởng (resume hiện có tự bỏ qua file còn nguyên).
- Web UI: chỉ thêm một liên kết tới trang HTML này trong danh sách artifact (`web/src/artifactPresentation.ts`); không xây giao diện mới.

### 10.3 Ngân sách và an toàn khi chạy dài
- Đếm số ảnh và clip đã sinh mỗi chương trong `render_status.json`; vượt `chapter_budget_*` → dừng chương đó, ghi ngoại lệ, chuyển chương sau.
- Thử lại lỗi mạng/provider dùng cơ chế retry hiện có; lỗi không thử lại được → ghi ngoại lệ cho shot, không dừng tập.

### 10.4 Dọn dẹp
- Chỉ khi người dùng xác nhận đã dùng bản mới ổn và không còn code nào gọi tới: xóa `agents/anime_director_guard.py`, `tests/test_anime_director_guard.py`, file allowlist trong `tests/series_name_allowlist.txt`.
- Cập nhật `configs/agent.example.yaml` và `HUONG_DAN_SU_DUNG.md` (mục mới: hồ sơ nhân vật, duyệt ngoại lệ).

## Test
- Clip rớt 2 lần → dùng Ken Burns, có ngoại lệ, tập vẫn ghép xong.
- Vượt ngân sách → chương dừng đúng chỗ, chương sau vẫn chạy.
- `apply_decisions` xóa đúng frame và chỉ shot đó được sinh lại.
