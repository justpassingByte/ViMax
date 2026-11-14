# Phase 06 — QA danh tính trên keyframe, làm lại trước khi tốn tiền video

Cờ: `quality.identity_qa`; thêm `quality.identity_margin` (mặc định 0.15), `quality.max_keyframe_retakes` (mặc định 2)

## Vì sao
- `agents/visual_qa.py` so keyframe với **panel** bằng một lượt VLM, mặc định chỉ báo cáo.
- Vòng làm lại đã có nhưng tắt mặc định (`manga_vision.auto_rerender_critical`, `pipelines/manga2video_pipeline.py` ~dòng 1000–1065). Vòng này chạy `Script2VideoPipeline` **trọn vẹn** (cả video) rồi mới QA, nên lỗi ngoại hình bị phát hiện sau khi đã trả tiền video.
- Không có kiểm tra bằng con số nào cho danh tính.

## Thay đổi

### 6.1 Module `agents/identity_qa.py`
```python
class FrameCheck(BaseModel):
    shot_idx: int; frame: Literal["first", "last"]
    expected: list[str]                 # character_id có mặt theo shot description
    detected: list[dict]                # {character_id|None, box, score}
    identity_scores: dict[str, float]   # cosine với tâm (form, outfit) đúng beat
    outfit_ok: dict[str, bool]          # VLM có/không so với mô tả trang phục ở beat
    has_lettering: bool                 # VLM có/không: bong bóng, chữ, khung truyện
    passed: bool; reasons: list[str]

async def check_frame(frame_path, shot, beat_id, profiles, image_embedder, vlm) -> FrameCheck
```
- Tìm nhân vật: hỏi VLM (chat model nhìn ảnh đã có) trả box + tên trong danh sách nhân vật có mặt. Không cần detector riêng ở phase này.
- Ngưỡng tự hiệu chỉnh, không cần nhãn: với mỗi nhân vật, tính độ giống trung bình giữa các ảnh mẫu của chính nhân vật đó (sheet + crop); gọi là `baseline`. Keyframe rớt danh tính nếu điểm < `baseline - identity_margin`.
- Rớt nếu: một nhân vật dự kiến rớt danh tính; số người khác dự kiến; trang phục sai; có chữ/bong bóng.
- Một lượt VLM cho cả ba câu hỏi (box, trang phục, chữ) mỗi frame để đỡ chi phí.

### 6.2 Chọn ứng viên tốt nhất
`generate_and_select_best_image` (`script2video_pipeline.py` ~dòng 1737): khi cờ bật và có embedder, chấm từng ứng viên bằng điểm danh tính trung bình; VLM chỉ phân xử khi chênh lệch < 0.02.

### 6.3 Thứ tự chạy mới trong `render_episode`
Khi `identity_qa: true` và `stop_after == "full"`:
1. `scene_pipe(stop_after="frames")`.
2. `check_frame` cho mọi keyframe. Shot rớt: xóa frame, chạy lại frames (seed mới). Tối đa `max_keyframe_retakes`. Lần cuối: đổi ảnh mẫu (góc khác hoặc crop nguồn khác của hồ sơ).
3. Vẫn rớt: giữ ứng viên có điểm cao nhất, ghi vào `scene_N/exceptions.json` (`{shot_idx, frame, reasons, best_score}`). Phase 10 gom lại.
4. `scene_pipe(stop_after="full")` dùng lại keyframe đã có.
- Ghi `scene_N/identity_qa.json` (mọi `FrameCheck`).
- Gộp vào `VisualQAReport` với `aspect` sẵn có (`character_identity`, `character_count`, `outfit_consistency`, `artifact`) để giao diện cũ vẫn đọc được.

## Test
- `check_frame` với embedder và VLM giả: điểm thấp → rớt đúng lý do; đúng hết → đạt.
- Vòng làm lại: shot rớt 2 lần rồi đạt → đúng 3 lần sinh; rớt mãi → có trong `exceptions.json`, pipeline vẫn chạy tiếp.
- Video chỉ được gọi sau khi keyframe đã qua QA (đếm số lần gọi video generator giả).
