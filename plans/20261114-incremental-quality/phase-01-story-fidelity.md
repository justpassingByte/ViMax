# Phase 01 — Bám sát cốt truyện: dọn prompt + cổng cốt truyện

Cờ: `quality.canon_gate`

## Vì sao
Prompt storyboard hiện **ra lệnh bịa thêm** nội dung, và không có bước nào chặn shot không dựa trên beat trước khi render:
- `agents/storyboard_artist.py`, `system_prompt_template_design_storyboard`:
  - ~dòng 65: mọi trận đánh phải thành chuỗi 5 bước "(1) incoming ambush … (4) desperate survival gambit …".
  - ~dòng 76: "enrich them with environmental B-roll".
  - ~dòng 110–118: "4-Phase Sakuga Grammar" (Pha 1–5) bắt buộc.
  - ~dòng 70–73 (Rescue Exclusivity), ~88 (Sky Streaks), ~92 (Outerwear Value/slate-grey coat): luật viết riêng cho một chương Eleceed.
  - ~dòng 94–99: "BAN EYE-LEVEL FLAT FRAMING".
- `agents/cinematic_director.py` có thể thêm shot `creative` (`environment_b_roll`, `aerial_reveal`…).
- `CinematicDirector.build_shot_provenance` chỉ **ghi** nguồn gốc shot, không chặn.
- Thoại trong `MangaDialogueLine.line` đã bị reader viết lại thành câu Nhật ≤ 12 từ; chỉ `vietsub` còn bám nghĩa gốc.

## Thay đổi

### 1.1 Prompt storyboard có căn cứ (không sửa prompt cũ)
- Thêm hằng `system_prompt_template_design_storyboard_grounded` trong `agents/storyboard_artist.py`. `design_storyboard` chọn template theo cờ.
- Bản mới **giữ**: định dạng JSON, luật thoại ≤ 12–15 từ, `[Speaker]/[Thought]` + `(Vietsub: …)` nguyên văn, không bong bóng/chữ trong khung, một tư thế mỗi frame, mô tả vị trí và hướng nhìn, transition và `target_duration_sec`.
- Bản mới **bỏ**: các khối liệt kê ở trên (5 bước đánh nhau, B-roll, 4-Phase Sakuga, Rescue Exclusivity, Sky Streaks, Outerwear Value, BAN EYE-LEVEL). Khối "Art Style Presets" theo mood giữ lại vì không ra lệnh bịa nội dung.
- Bản mới **thêm luật căn cứ**:
  1. Mỗi shot bắt đầu `visual_desc` bằng `[RefBeat: <beat_id>]` của một beat **trong script cảnh này**.
  2. Thứ tự beat được dẫn không giảm (trừ shot có tag `[Flashback]`).
  3. Chỉ dùng thoại có sẵn trong script; không thêm câu mới; mỗi câu xuất hiện đúng một lần.
  4. Chỉ đưa vào khung nhân vật có trong beat được dẫn.
  5. Được thêm shot phụ (toàn cảnh, phản ứng, đặc tả) nhưng mỗi shot phụ phải dẫn beat nó minh họa và tổng shot phụ ≤ 25% số shot.
  6. Sáng tạo được phép: góc máy, ánh sáng, biểu cảm, chuyển động, hiệu ứng, chuyển cảnh. Không được: sự kiện mới, tương tác mới, đổi thứ tự.
- Prompt "critic" trong `review_and_repair_storyboard` (~dòng 330): khi cờ bật, thêm luật số 8 "không thêm shot, nhân vật hay thoại không có trong SCRIPT; giữ `[RefBeat]`".

### 1.2 Module mới `agents/canon_gate.py` (thuần code, không LLM)
```python
class CanonIssue(BaseModel):
    shot_idx: int
    kind: Literal["uncited", "unknown_beat", "invented_dialogue", "missing_dialogue",
                  "duplicate_dialogue", "foreign_character", "order_violation", "coverage_ratio"]
    detail: str

class CanonReport(BaseModel):
    scene_idx: int
    passed: bool
    issues: list[CanonIssue]
    stats: dict  # beat_coverage, dialogue_fidelity, coverage_shot_ratio

def check_storyboard(storyboard, beats, characters) -> CanonReport: ...
def repair_storyboard(storyboard, beats, report) -> list[ShotBriefDescription]: ...
```
- Đọc beat id bằng `_clean_refbeat_tags`/`extract_shot_beat_refs` đã có.
- So thoại: chuẩn hóa (bỏ dấu câu, lowercase), so với `line` **và** `vietsub` của beat bằng `difflib.SequenceMatcher` ≥ 0.9. Dùng lại parser thoại trong `utils/audio.py` (`extract_all_dialogues_from_shot`).
- Nhân vật trong shot: các tag `<Tên>` trong `visual_desc` so với `characters_present` của beat được dẫn (qua alias map của roster).
- `repair_storyboard` (cố định, không LLM): bỏ shot `uncited`/`unknown_beat`/`invented_dialogue`/`foreign_character`; với mỗi câu `missing_dialogue` chèn một shot đơn giản (cỡ trung, người nói, dẫn đúng beat, giữ nguyên `[Speaker]` + `(Vietsub)`); sắp lại theo thứ tự beat (ổn định); đánh lại `idx`.

### 1.3 Tích hợp vào `pipelines/manga2video_pipeline.py::plan_episode`
Vị trí: sau khối `lint_and_auto_correct_storyboard` + `review_and_repair_storyboard` (~dòng 760–777), **trước** `bind_storyboard_to_panel_anchors` (hàm này xóa tag `[RefBeat]`).
```
for attempt in range(2):
    report = check_storyboard(...)
    if report.passed: break
    storyboard = await script_pipe.design_storyboard(..., user_requirement=extra_req + failure_list(report))
    # lưu ý: design_storyboard có thể đọc lại storyboard.json đã cache — phải bỏ qua cache khi thử lại
if not report.passed:
    storyboard = repair_storyboard(storyboard, scene_beats, report)
    report = check_storyboard(...)  # phải passed
write scene_dir/canon_report.json
```
Khi cờ tắt: không chạy gì mới.

### 1.4 Giới hạn cinematic director (khi cờ bật)
- `agents/cinematic_director.py::plan_scene_expansion`: chỉ giữ shot_type `establishing_wide` (chỉ ở beat đầu cảnh hoặc beat `is_scene_break`), `reaction_close_up`, `insert_detail`, `over_the_shoulder`, `alternate_angle`; bỏ `environment_b_roll`, `aerial_reveal`, `return_to_panel`.
- Tổng shot thêm của cảnh ≤ 25% số beat (cắt theo `confidence` giảm dần).

### 1.5 Giữ câu gốc
- Thêm trường `source_text: str = ""` vào `MangaDialogueLine` (`interfaces/manga.py`): chữ gốc trong bong bóng. Thêm một dòng vào prompt reader yêu cầu điền.
- Cache cũ không có trường này → canon gate dùng `vietsub` + `line` như trên; không bắt buộc đọc lại chương.

## Test
- `tests/test_canon_gate.py`: storyboard giả có 1 shot không tag, 1 câu thoại bịa, 1 nhân vật lạ, 1 beat đảo thứ tự, 1 câu thoại bị thiếu → `check_storyboard` báo đủ 5 loại; `repair_storyboard` cho kết quả `passed`.
- Test chọn template theo cờ.
- Test tích hợp với chat model giả (đã có kiểu fake trong `tests/test_manga2video_pipeline.py`): lần 1 trả storyboard lỗi, lần 2 trả đúng → chỉ 2 lần gọi, `canon_report.json` passed.

## Không làm
Không thêm bộ lên shot (phase 08), không đụng guard (phase 02), không dùng LLM để kiểm sự kiện canon (phase 09).
