# Phase 08 — Bộ lên shot theo luật

Cờ: `quality.shot_planner` (cần `canon_gate` và `generic_linter` đã bật)

## Vì sao
Góc máy lặp lại vì guard đổi mọi medium/eye-level thành góc thấp và prompt cấm góc ngang tầm mắt (đã gỡ ở phase 01–02). Phase này thêm phần còn thiếu: chọn cỡ cảnh/góc/chuyển động **theo loại beat**, có luật đa dạng, không cần agent đạo diễn và không thêm lượt gọi LLM.

## Thay đổi

### 8.1 Loại beat
- Thêm `beat_type: Literal["establishing", "dialogue", "exchange", "reaction", "action", "impact", "reveal", "monologue", "transition"] | None = None` vào `MangaBeat` (`interfaces/manga.py`).
- Prompt reader: thêm một dòng yêu cầu điền `beat_type`.
- Với cache cũ (không có trường): hàm `infer_beat_type(beat, prev_beat)` thuần code trong `agents/shot_planner.py`:
  - `is_scene_break` hoặc beat đầu cảnh không có thoại → `establishing`
  - `line_type` là thought/narration → `monologue`
  - có thoại và người nói khác beat trước nhưng cùng cảnh → `exchange`; có thoại → `dialogue`
  - có `sfx` + từ khóa hành động trong `action_desc` → `action`; sfx mạnh (CRASH, BOOM…) → `impact`
  - không thoại, chỉ biểu cảm → `reaction`
  - còn lại → `transition`
  Từ khóa để ở `configs/shot_rules.yaml`, không viết cứng trong code.

### 8.2 Bảng luật `configs/shot_rules.yaml`
Mỗi loại beat: danh sách cỡ cảnh, góc, chuyển động, hiệu ứng được phép, kèm trọng số. Nội dung đúng như bảng "Lên góc máy" trong doc kế hoạch. Luật đa dạng:
```yaml
variety:
  max_same_size_angle_run: 2
  establishing_on_location_change: true
  dialogue_reverse_after_lines: 4      # >4 câu: phải có góc ngược + 1 two-shot
  min_distinct_sizes_if_shots_ge: {shots: 6, sizes: 3}
  max_low_angle_share_non_action: 0.30
  fx_allowed_beat_types: [action, impact, reveal, monologue]
  max_coverage_ratio: 0.25
```

### 8.3 Module `agents/shot_planner.py`
```python
class ShotSlot(BaseModel):
    beat_ids: list[str]; beat_type: str
    shot_size: str; camera_angle: str; camera_movement: str
    fx: list[str] = []; is_coverage: bool = False
def plan_scene(beats, rules, seed: int) -> list[ShotSlot]
def check_variety(slots_or_storyboard, rules) -> list[str]   # danh sách vi phạm
```
- Chọn ngẫu nhiên có trọng số với seed cố định theo cảnh (lặp lại được), quay lui đơn giản khi vi phạm luật.
- Shot phụ: tối đa `max_coverage_ratio`, chỉ là `establishing` (đầu cảnh khi đổi địa điểm) hoặc `reaction` (sau beat có thoại quan trọng), luôn dẫn beat có thật.

### 8.4 Đấu nối
- `ShotBriefDescription` (`interfaces/shot_description.py`): thêm trường tùy chọn `beat_ids`, `beat_type`, `shot_size`, `camera_angle`, `camera_movement`, `fx`, `characters`. Mặc định rỗng, dữ liệu cũ vẫn đọc được.
- `plan_episode`: trước `design_storyboard`, gọi `plan_scene`; đưa khối `SHOT SLOTS` vào prompt grounded (phase 01): "Shot k: [RefBeat: …] size=…, angle=…, movement=…; viết mô tả trong khuôn này". Số shot = số slot.
- Sau LLM: ghi đè các trường bằng giá trị slot; nếu `visual_desc` không nhắc cỡ cảnh/góc thì thêm câu mở đầu cố định ("Medium close-up, eye level, static camera."). Chạy `check_variety`; vi phạm → ghi vào `canon_report.json` (không chặn).
- Hiệu ứng: ánh xạ `fx` sang tag `[AnimeFX: …]` mà `utils/video.py::resolve_and_apply_anime_fx` đã hiểu (impact → `Impact`, speed_lines → `SpeedLines`, shake → `ScreenShake`, flashback → `Flashback`, dramatic_focus → `DramaticFocus`, white_flash → `transition_in: white_flash`). Xóa tag `[AnimeFX: …]` do LLM tự thêm nếu loại beat không cho phép.
- `cam_idx`: các shot cùng cỡ+góc+cùng tập nhân vật trong một cảnh dùng chung `cam_idx`; còn lại mỗi shot một camera. Không đổi `construct_camera_tree`.

## Test
- `infer_beat_type` với các beat mẫu.
- `plan_scene` lặp lại được với cùng seed; không vi phạm luật với cảnh 3, 8, 20 beat.
- Ghi đè trường và ánh xạ hiệu ứng đúng; tag hiệu ứng bị cấm bị xóa.
