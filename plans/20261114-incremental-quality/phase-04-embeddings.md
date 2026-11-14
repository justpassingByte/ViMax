# Phase 04 — Module embedding (model miễn phí trên Hugging Face)

Cờ: `quality.embeddings` (`off` | `dinov2`, mặc định `dinov2`), `quality.text_embeddings` (`off` | `e5`, mặc định `e5`), `quality.scene_split_threshold`

## Vì sao
Hiện không có embedding ảnh nào. `agents/identity_dedup.py` embed **mô tả chữ**, rơi về bản giả lập băm token (`MockHashingEmbedder`) khi không có provider, và chỉ báo cáo. Ghép cảnh trong `agents/manga_reader.py::build_episode_scenes` so `setting` theo đúng từng chữ → 54 cảnh cho ch.1; `world_model_builder` gộp địa điểm bằng Jaccard token → 31 địa điểm.

Embedding chỉ giúp **chọn và kiểm** ảnh mẫu; không làm model ảnh làm theo ảnh mẫu.

## Thay đổi

### 4.1 Phụ thuộc tùy chọn
- Thêm nhóm `[dependency-groups] embeddings = ["torch", "transformers", "numpy"]` trong `pyproject.toml` (torch đã có index cu128). Code import lười; thiếu thư viện → cờ tự về `off` kèm cảnh báo, pipeline vẫn chạy.
- **Ổ đĩa:** máy người dùng thiếu chỗ trống ở ổ C. **Hỏi người dùng đường dẫn ổ đĩa** rồi đặt `HF_HOME` sang đó trước khi tải model. Chỉ tải 2 model: dinov2-small (~90 MB) và multilingual-e5-small (~470 MB). `torch` bản CUDA ~3 GB.
- **License:** trước khi dùng, mở trang model trên Hugging Face, ghi license vào docstring của `tools/embeddings.py`. Không dùng model có license phi thương mại.

### 4.2 Module `tools/embeddings.py`
```python
class ImageEmbedder(Protocol):
    name: str
    def embed_images(self, paths: list[str]) -> np.ndarray: ...   # L2-normalized, shape (n, d)
class TextEmbedder(Protocol):
    name: str
    def embed_texts(self, texts: list[str]) -> np.ndarray: ...

def build_image_embedder(kind: str, device: str = "auto") -> ImageEmbedder | None
def build_text_embedder(kind: str, device: str = "auto") -> TextEmbedder | None
```
- Model: `facebook/dinov2-small` (ảnh) và `intfloat/multilingual-e5-small` (chữ). Mở trang model để xác nhận license trước khi dùng.
- Cache vector theo sha1 nội dung ảnh/chữ vào `<working_dir>/embeddings/<model>/`. Chạy được trên CPU; dùng CUDA nếu có.
- Không gọi mạng ngoài lúc chạy pipeline ngoài lần tải model đầu tiên.

### 4.3 Dùng embedding ở 3 chỗ (đều sau cờ)
1. **Làm sạch seed** — hàm mới `agents/identity_dedup.py::audit_seed_crops(seed_refs, panel_dir, embedder)`:
   - Trùng hash giữa các nhân vật; trùng hash với một panel nguyên trong `panel_anchors/`.
   - Crop gần ≥ 2 tâm nhân vật (cosine trong khoảng ε của nhau) → `ambiguous`.
   - Ghi `character_seed_audit.json`; khi cờ bật, loại seed lỗi khỏi `character_seed_refs.json` trước khi sinh portrait (`manga_reader.build_character_roster_and_seeds`).
2. **Trùng danh tính** — `detect_duplicate_identities` nhận thêm `image_embedder`: điểm = trung bình cosine tâm ảnh + độ giống alias + cờ "không bao giờ cùng panel". Vẫn chỉ đề xuất (phase 05 dùng để ghép).
3. **Gom cảnh và địa điểm** — với `text_embeddings: e5`:
   - `build_episode_scenes`: thay so sánh chuỗi `setting` bằng cosine của `setting` hai beat liên tiếp < ngưỡng (mặc định 0.80, chỉnh được trong config `quality.scene_split_threshold`) hoặc `is_scene_break`.
   - `world_model_builder._merge_environments`: gộp theo cosine thay Jaccard; bỏ các "địa điểm" là nền trắng (`plain white`, `no location`) bằng cách gán vào địa điểm của beat kề trước.

## Test
- Embedder giả (vector cố định) cho mọi test; không tải model trong CI.
- `audit_seed_crops` bắt đúng: seed trùng giữa 2 nhân vật, seed trùng panel.
- Gom cảnh bằng embedding giả: chuỗi setting na ná nhau không tách cảnh.
