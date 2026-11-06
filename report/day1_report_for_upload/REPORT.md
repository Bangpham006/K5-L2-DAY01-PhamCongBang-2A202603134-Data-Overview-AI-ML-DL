# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.10 / PyTorch 2.5 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   - Nói về cả bức ảnh. Vì đây là tác vụ phân loại ảnh (image classification), mô hình gán một nhãn chung đại diện cho toàn bộ khung hình chứ không định vị riêng lẻ từng vật thể.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   - Đi kèm taxonomy/checkpoint. Cụ thể là danh sách 1000 lớp của ImageNet-1K có sẵn từ file checkpoint `yolo11n-cls.pt`, mô hình không tự nghĩ ra tên lớp.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   - `class_id` là mã số để lập trình và lưu trữ không bị nhầm; `class_name` để người đọc hiểu; `taxonomy_name` để biết nhãn thuộc bộ quy chuẩn nào để đối chiếu và tái lập kết quả.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   - Guideline cần có quy tắc ưu tiên: chọn chủ thể chiếm diện tích lớn nhất hoặc nằm ở trung tâm ảnh, và quy định rõ thứ tự ưu tiên nếu có nhiều loại vật thể cùng xuất hiện.
6. Vì sao không được lấy `score` của model làm ground truth?
   - Vì `score` chỉ là độ tự tin toán học lúc mô hình dự đoán, mô hình vẫn có thể đoán sai dù score cao. Ground truth phải do người kiểm tra và xác nhận theo quy chuẩn.

## 2. Phát hiện vật thể

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   - Góc trên-trái nằm tại tọa độ pixel (385.33, 69.24), góc dưới-phải nằm tại pixel (498.92, 348.92), chiều rộng 113.58 pixel và chiều cao 279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   - Ngưỡng 0.20: 17 prediction.
   - Ngưỡng 0.60: 6 prediction.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   - Reviewer phải kiểm tra và xóa bỏ nhiều hộp dự đoán sai (false positive), hộp trùng lặp hoặc hộp vẽ lệch do mô hình nhận diện bừa ở độ tin cậy thấp.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   - Bounding box phải vẽ sát mép ngoài của vật thể (tight box), ôm trọn toàn bộ phần nhìn thấy được, không để thừa khoảng trống quá mức và không cắt lẹm vào vật thể.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   - Cái bát (`bowl`) phía sau bị bàn bếp che khuất một phần. Nếu bị che khuất thông thường (vẫn thấy rõ hình dạng) thì đưa vào guideline (vẽ box phần nhìn thấy và gắn nhãn che khuất); nếu bị che gần hết hoặc quá mờ thì dùng escalation hỏi Lab Coach để thống nhất tiêu chuẩn chung.

## 3. Phân đoạn theo từng đối tượng


1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   - Polygon thể hiện đúng hình dạng đường viền thực tế của vật thể theo từng pixel và loại bỏ được phần nền thừa bên trong hình chữ nhật.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   - `instance_id` dùng để phân biệt từng cá thể riêng biệt trong cùng 1 bức ảnh (ví dụ người 1 khác người 2). Nó không phải `class_id` vì không đại diện cho tên loài, và không phải `tracking_id` vì chỉ có tác dụng trong ảnh đơn lẻ này, không theo dõi đối tượng qua video.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   - Các điểm polygon phải bám sát theo mép viền thực của vật thể, bo mượt các đoạn cong, không để mask lấn ra nền hoặc lấn sang vật thể kế bên.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   - Chỗ tiếp xúc giữa tay người (`person`) và mặt bàn hoặc cái bát (`bowl`). Dùng guideline nếu là tiếp xúc thông thường (ưu tiên vẽ đủ vật nằm trước, vật phía sau vẽ bám sát viền không đè lên); chỉ dùng escalation nếu ranh giới quá tối hoặc mờ không phân biệt được bằng mắt.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn lớp duy nhất cho toàn bức ảnh. | Ảnh `traffic` có nhiều loại xe (bus, cab, car) cùng xuất hiện, model đoán `cab` score chỉ 0.51, khó chọn 1 nhãn duy nhất. | Chọn đúng chủ thể chính chiếm diện tích lớn nhất theo quy tắc ưu tiên trong guideline. | Kiểm tra nhãn được chọn có đúng quy tắc ưu tiên không, có chọn nhầm sang vật thể phụ không. |
| Phát hiện vật thể | Tập hợp các cặp nhãn và hộp chữ nhật `[class_id, bbox_xyxy]` cho từng vật. | Ảnh `kitchen` có các đồ vật nhỏ hoặc bị che khuất (như thìa, cốc, bát phía sau) dễ bị sót hoặc vẽ box lệch. | Vẽ box bám sát từng vật nhìn thấy, gán đúng lớp và đánh dấu trạng thái che khuất nếu có. | Soát các vật bị bỏ sót, kiểm tra độ khít của box và xóa các box thừa hoặc vẽ sai lớp. |
| Instance segmentation | Tập hợp các cặp nhãn, đa giác và mã cá thể `[class_id, polygon_xy, instance_id]`. | Ranh giới phức tạp ở vùng tiếp xúc giữa người và bàn/bát trong ảnh `kitchen`. | Chấm các điểm polygon bám sát đường viền thực của từng đối tượng, tách riêng từng instance. | Kiểm tra mép viền polygon có ăn vào nền không, có cắt lẹm vào vật thể không và đảm bảo không gộp 2 vật vào chung 1 mask. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   - Chỉ dùng các ảnh COCO mẫu được cấp trong bài lab, tuyệt đối không tải ảnh cá nhân, khuôn mặt, biển số xe hay dữ liệu nhạy cảm lên Colab và GitHub.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   - Dừng làm bài ngay và báo cho Giảng viên hoặc Lab Coach phụ trách để được xử lý.

## 6. Checklist trước khi nộp link repo



- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
