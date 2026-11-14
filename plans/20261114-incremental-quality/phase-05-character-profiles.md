# Phase 05 — Hồ sơ nhân vật và chọn ảnh mẫu theo trạng thái

Cờ: `quality.profiles`, `quality.ref_policy: profile`

## Vì sao
- Mỗi nhân vật chỉ có **một** bộ portrait cho cả bộ truyện (`character_portraits_registry.json`), không có biến thể trang phục/trạng thái.
- Ở chế độ series, `_merge_series_bank` (`pipelines/manga2video_pipeline.py` ~dòng 1244) ghép nhân vật theo **đúng chuỗi tên** và chỉ giữ 4 seed **đầu tiên** của một tên (`if seeds and name not in series_seed_refs`) → đổi đồ ở chương sau không bao giờ được cập nhật; "Kayden" và "Kayden (Human)" thành hai người.
- Thông tin danh tính nằm rải rác ở `characters.json`, `character_seed_refs.json`, registry portrait và (sau phase 02) `series/<slug>/characters.yaml`.

## Thay đổi

### 5.1 Schema `interfaces/profile.py`
```python
class OutfitSheet(BaseModel): front: str = ""; side: str = ""; back: str = ""
class Outfit(BaseModel):
    outfit_id: str; label: str; description: str
    valid_from_beat: str; valid_to_beat: str | None = None   # "ch01_b012"; so sánh được theo (chương, số beat)
    sheet: OutfitSheet = OutfitSheet()
class StateEntry(BaseModel):
    kind: Literal["injury", "item", "hair", "form", "other"]; text: str
    from_beat: str; to_beat: str | None = None; evidence_beat_ids: list[str] = []
class SourceCrop(BaseModel): path: str; beat_id: str; form_id: str = ""; outfit_id: str = ""; usable: bool = True
class VoiceProfile(BaseModel):          # phase 07 điền
    mode: str = "native"; description: str = ""; reference_audio: str = ""; engine: str = ""; voice_id: str = ""; language: str = "ja"
class CharacterProfile(BaseModel):
    character_id: str; display_name: str; aliases: list[str] = []
    forms: list[dict] = []              # {form_id, label, species}
    permanent_features: str = ""
    outfits: list[Outfit] = []; states: list[StateEntry] = []
    source_crops: list[SourceCrop] = []
    embedding_paths: dict[str, str] = {}  # key "<form_id>/<outfit_id>" -> .npy
    voice: VoiceProfile = VoiceProfile()
    prompt_notes: str = ""              # từ series/<slug>/characters.yaml
    status: Literal["draft", "approved", "locked"] = "draft"; version: int = 1
```

### 5.2 Kho hồ sơ `utils/profile_store.py`
- Vị trí: cạnh `series_characters_path` của series; chạy một tập thì `<working_dir>/profiles/`. Mỗi nhân vật `profiles/<character_id>/profile.json` + thư mục `sheets/`, `crops/`.
- API: `load_all()`, `get(name_or_alias)`, `upsert(profile)`, `outfit_at(character_id, beat_id)`, `states_at(character_id, beat_id)`, `add_crops(...)`, `match(candidate) -> (profile_id | None, score, decision)`.
- Không bao giờ ghi đè hồ sơ `locked` trừ trường `source_crops` (thêm) và `states` (thêm bản ghi mới có bằng chứng).

### 5.3 Dựng và cập nhật hồ sơ (sau bước roster của reader)
Trong `plan_episode`, ngay sau `build_character_roster_and_seeds`:
1. Mỗi nhân vật trong roster → `profile_store.match`: alias khớp chính xác → cùng hồ sơ; nếu không thì điểm phase 04 (tâm ảnh + alias + không cùng panel): ≥ ngưỡng cao → ghép tự động (thêm alias); vùng giữa → ghi `profile_merge_suggestions.json` và tạm tạo hồ sơ riêng; thấp → hồ sơ mới `draft`.
2. Crop: từ `characters_observed` của các chunk, đã qua `audit_seed_crops` (phase 04) → `source_crops`.
3. Trang phục: gom `outfit_summary` của các quan sát cùng nhân vật theo text embedding; cụm mới có ≥ 2 quan sát → `Outfit` mới, `valid_from_beat` = beat đầu tiên của cụm; đóng `valid_to_beat` của trang phục trước.
4. Chỉ sinh sheet cho **dàn chính** (xuất hiện ≥ `quality.profile_min_appearances`, mặc định 5 beat) để kiểm soát chi phí: dùng `CharacterPortraitsGenerator` hiện có, đưa mô tả trang phục + 2–3 crop tốt nhất của cụm đó.
5. Tính embedding tâm cho mỗi (form, outfit).
6. `prompt_notes` lấy từ `series/<slug>/characters.yaml` nếu có.
- Thay `_merge_series_bank` bằng `profile_store` khi cờ bật. Vẫn ghi `characters.json` như cũ để các bước sau không phải đổi.

### 5.4 Chọn ảnh mẫu `ref_policy: profile`
Ở các chỗ dựng ứng viên ảnh mẫu trong `script2video_pipeline.py` (đã gom về một hàm ở phase 03):
- Xác định beat của shot từ `shot_panel_refs.json` / `shot_provenance.json`.
- Mỗi nhân vật có mặt: hình dạng (theo tên form), trang phục `outfit_at(beat)`, góc nhìn theo từ khóa trong `ff_desc` → 1 ảnh sheet. Nếu còn chỗ: 1 crop nguồn giống nhất (cosine) với crop panel của beat đó.
- Trạng thái `states_at(beat)` (vết thương, đồ cầm) → thêm một câu vào prompt.
- `ReferenceImageSelector` chỉ còn viết prompt cho các ảnh đã chọn (không chọn ảnh nữa). Ghi `*_refs_used.json` như phase 03.

### 5.5 Duyệt nhanh
- `tools/profile_sheet.py`: sinh HTML một trang cho mỗi bộ truyện: mỗi hồ sơ một hàng (sheet theo trang phục, 6 crop, alias, gợi ý ghép). Người dùng duyệt dàn chính một lần.
- CLI `python -m utils.profile_store approve|lock|merge <id> [<id2>]`.
- Pipeline vẫn chạy với hồ sơ `draft` (để không chặn), nhưng ghi cảnh báo.

## Test
- `outfit_at` đúng ranh giới beat qua nhiều chương.
- `match`: alias khớp, ghép tự động, vùng giữa → suggestion, không ghép hai tên từng cùng một panel.
- Hồ sơ `locked` không bị ghi đè.
- Chọn ảnh mẫu: shot ở beat sau khi đổi đồ nhận sheet trang phục mới.
