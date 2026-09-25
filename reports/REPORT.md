# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Bùi Công Mạnh

Công cụ gán nhãn đã dùng: CVAT (xuất định dạng "Ultralytics YOLO Detection 1.0")

## 1. Dữ liệu và cách chia tập

Pool (tập chưa gán nhãn) và test set được chia theo trục thời gian, có vùng đệm ở giữa, thay vì
chia ngẫu nhiên, vì dữ liệu là các khung hình trích ra liên tục từ một đoạn video giao thông cao
tốc ban đêm. Các khung hình đứng gần nhau về thời gian gần như trùng lặp cảnh: ví dụ trong
`selection_round1.csv`, `frame_0369.jpg` (t=147.6s, được chọn) và `frame_0372.jpg` (t=148.8s, bị
loại) chỉ cách nhau 1.2 giây và gần như cùng một đoạn giao thông.

Nếu chia ngẫu nhiên, nhiều khung gần trùng của cùng một xe/cùng một đoạn cảnh sẽ rơi vào cả tập
train và tập test (rò rỉ dữ liệu theo thời gian — temporal leakage). Khi đó mô hình coi như đã
"nhìn thấy" gần như nguyên vẹn cảnh đó lúc train, nên số đo trên test (AP50, recall...) sẽ bị lệch
theo hướng lạc quan giả tạo — cao hơn khả năng thực sự của mô hình khi gặp giao thông hoàn toàn mới.
Vùng đệm (buffer) ở giữa pool và test giúp đảm bảo không có khung hình nào ở biên hai tập là gần
trùng nhau, giữ cho test set thực sự đại diện cho dữ liệu "tương lai chưa từng thấy".

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0 từ `rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Test set: 20 ảnh, 403 box tham chiếu (bỏ qua 14 box cao dưới 16 px), IoU 0.5, conf 0.25.
TP=197, FP=16, FN=206 (`metrics_round0.json`). Guideline chấm điểm (`HUONG_DAN.txt`) xác nhận: chỉ
có một lớp `car` (mọi xe từ 4 bánh trở lên), box ôm sát thân xe, không tính vệt sáng đèn pha, và xe
quá xa (box cao dưới ~16 px) không được tính điểm — khớp với số 14 box bị bỏ qua ở trên.

Dựa vào `outputs/compare_round0.jpg` (cột phải là cold start so với nhãn tham chiếu bên trái), mô
hình khởi đầu lạnh không khớp nhãn tham chiếu chủ yếu ở:

- **Xe ở xa, thân xe nhỏ**: recall theo kích thước cho thấy recall xe nhỏ chỉ 18.2% (12/66 box),
  thấp hơn rất nhiều so với xe vừa (54.7%) và xe lớn (56.1%). Mô hình COCO gốc không được huấn
  luyện nhiều trên xe nhỏ, xa, chỉ thấy vài chục pixel trong ảnh cao tốc ban đêm.
- **Xe bị lóa đèn pha hoặc ngược sáng**: nhiều khung hình có xe chỉ hiện ra dưới dạng cụm đèn pha
  sáng chói, viền thân xe gần như biến mất trong glare — đây là nguồn FN chính (206 box bị bỏ sót).
  Quan sát Blind Scan độc lập trên `frame_0099.jpg` (xem `BLIND_SCAN.md`) xác nhận đúng hai điểm
  yếu này: xe ở góc/rìa khung hình và xe có đèn mờ là hai vị trí dễ bị bỏ sót nhất — khớp với 8/8
  box bị AI bỏ sót trên ảnh đó khi đối chiếu với nhãn cuối cùng (`round1_diff.md`).
- Một số ít FP (16 box) xuất hiện ở vệt sáng đèn đường/phản chiếu mặt đường bị nhận nhầm thành xe —
  đúng loại lỗi mà `HUONG_DAN.txt` cảnh báo ("box ôm vệt đèn" là lỗi cần xoá khi rà nhãn), và cũng
  đúng loại lỗi mình gặp khi rà `frame_0312.jpg`: AI vẽ 1 box ở vùng phải-giữa khung hình không khớp
  xe thật (nghi là phản chiếu ánh sáng), mình đã xoá box này (xem `REVIEW_LOG.csv`).

Độ phủ theo kích thước xe cho thấy mô hình cold start lệch mạnh về phát hiện xe lớn/vừa (ở gần
camera, sáng rõ) và gần như "mù" với xe nhỏ/xa — đúng như kỳ vọng với một mô hình tổng quát chưa
từng thấy đặc thù camera giao thông ban đêm này.

Một trường hợp cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai: `frame_0326.jpg`,
nơi AI vẽ một box hơi lệch/to hơn thân xe thật (IoU với box đã sửa chỉ ~0.52) — nếu chỉ tính theo
IoU tự động mà không nhìn lại ảnh, đây dễ bị coi là "AI sai hoàn toàn", nhưng thực tế AI đã định vị
đúng xe, chỉ cần chỉnh nhẹ ranh giới box, không phải model hoàn toàn bỏ sót.

## 3. Chiến lược chọn mẫu

Công thức `score = W_U·U + W_A·A + W_D·D` kết hợp ba tín hiệu, mỗi tín hiệu chuẩn hoá về [0, 1]:

- **U (uncertainty — độ bất định)**: đo mức độ mô hình hiện tại "phân vân" khi dự đoán trên ảnh đó.
  Theo `to_label/round1/batch.json`, lô này được đề xuất bởi model cold start
  (`"proposed_by": "yolov8n cold start (COCO car+bus+truck)"`), dùng chiến lược `"strategy":
  "uncertainty"`. Ảnh có U cao là ảnh mô hình dễ sai nhất.
- **A (ambiguity — độ mơ hồ khi gán nhãn)**: tỉ lệ số box khó xác định ranh giới (n_ambiguous) trên
  tổng số box, chuẩn hoá theo ngưỡng cố định (giới hạn tối đa 1.0). Nhiều ảnh trong top của
  `selection_round1.csv` có A=1.0 (ví dụ `frame_0182.jpg`, `frame_0312.jpg`) vì cảnh quá đông xe,
  gần một nửa số box là ca khó.
- **D (gần trùng/diversity theo thời gian)**: phạt các khung hình đứng quá gần một khung đã chọn về
  mặt thời gian. `batch.json` xác nhận `"min_gap_s": 2.0` — khoảng cách thời gian tối thiểu bắt
  buộc giữa các khung được chọn là 2 giây.

Bằng chứng cụ thể từ `outputs/selection_round1.csv` cho vai trò của `MIN_GAP_S`: `frame_0369.jpg`
(hạng 2, t=147.6s, score=0.9324, **selected=True**) đã được chọn, khiến hai khung rất gần nó về
thời gian bị loại dù điểm không hề thấp: `frame_0372.jpg` (hạng 6, t=148.8s, cách 1.2s, score=0.9101
— cao hơn cả `frame_0107.jpg` hạng 14 đã được chọn — nhưng **selected=False**) và `frame_0368.jpg`
(hạng 9, t=147.2s, cách 0.4s, score=0.9003, cũng **selected=False**). Đây là ví dụ rõ nhất cho thấy
hệ thống ưu tiên đa dạng theo thời gian hơn điểm số tuyệt đối.

Ba frame thuộc lô 12 ảnh model chọn (`selected=True` trong `selection_round1.csv`, khớp
`to_label/round1/batch.json` và `outputs/selection_round1.jpg`): `frame_0182.jpg` (hạng 1, score
0.9591, A=1.0, D=1.0 — điểm cao nhất, gần một nửa số box là ca khó), `frame_0331.jpg` (hạng 5,
score 0.9154, n_boxes=47 — cảnh đông xe nhất trong lô), `frame_0099.jpg` (hạng 8, score 0.9063,
chính là ảnh mình đã Blind Scan — 13/13 box AI đều đúng vị trí, nhưng bỏ sót thêm 8 xe khác).

Điểm bất định (U) không tự chứng minh ảnh đó sẽ cải thiện mô hình: nó chỉ phản ánh mô hình hiện tại
đang "phân vân" ở đâu, không đảm bảo việc gán đúng nhãn cho ảnh đó sẽ giúp mô hình tổng quát hoá tốt
hơn trên toàn bộ pool. Thực tế đã kiểm chứng: dù lô 12 ảnh vòng 1 được chọn kỹ theo U/A/D và được rà
nhãn rất kỹ (167/169 box AI giữ nguyên, 51 box thêm mới — xem `round1_diff.md`), AP50 sau khi train
vẫn giảm mạnh (0.771 → 0.420, xem mục 4) — chứng minh điểm bất định cao không đảm bảo model học tốt
hơn, vì kết quả huấn luyện còn phụ thuộc nhiều yếu tố khác (learning rate, số epoch, số lượng ảnh).

## 4. Các vòng học chủ động (active learning)

Bảng từ `rounds_table.md` (test set cố định: 20 ảnh, 403 box tham chiếu, IoU 0.5, conf 0.25):

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 177 | 0.420 | **-0.351** | 1.000 | 0.017 | 0.034 | 0.000 | 0.017 | 0.049 |

**Vòng 1** (12 ảnh: frame_0099, 0107, 0182, 0187, 0227, 0270, 0312, 0326, 0331, 0369, 0380, 0392;
177 box theo `metrics_round1.json`, 50 epoch, chiến lược `uncertainty`): AP50 giảm mạnh từ 0.771
xuống 0.420 (Δ = -0.351). TP giảm từ 197 → 7, FP giảm từ 16 → 0, FN tăng từ 206 → 396. Precision
đạt tuyệt đối 1.0 nhưng recall sụp xuống còn 1.7%.

**Mức độ sửa nhãn gợi ý** (đối chiếu `to_label/round1/prelabels/*.txt` với nhãn CVAT cuối cùng,
xem bảng đầy đủ trong `round1_diff.md`):

| frame | box AI | box cuối | giữ nguyên | chỉnh sửa | xoá | thêm mới |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frame_0099 | 13 | 21 | 13 | 0 | 0 | 8 |
| frame_0107 | 13 | 20 | 13 | 0 | 0 | 7 |
| frame_0182 | 13 | 18 | 13 | 0 | 0 | 5 |
| frame_0187 | 14 | 20 | 14 | 0 | 0 | 6 |
| frame_0227 | 13 | 13 | 13 | 0 | 0 | 0 |
| frame_0270 | 13 | 15 | 13 | 0 | 0 | 2 |
| frame_0312 | 13 | 18 | 12 | 0 | 1 | 6 |
| frame_0326 | 15 | 20 | 14 | 1 | 0 | 5 |
| frame_0331 | 20 | 22 | 20 | 0 | 0 | 2 |
| frame_0369 | 14 | 20 | 14 | 0 | 0 | 6 |
| frame_0380 | 15 | 16 | 15 | 0 | 0 | 1 |
| frame_0392 | 13 | 16 | 13 | 0 | 0 | 3 |
| **Tổng** | **169** | **219** | **167** | **1** | **1** | **51** |

Gần như toàn bộ box AI đề xuất được giữ nguyên (167/169, 98.8%), chỉ 1 box bị xoá và 1 box bị chỉnh
sửa — AI hiếm khi vẽ sai hoàn toàn, lỗi chủ yếu là **bỏ sót xe** (51 box thêm mới, khớp Blind Scan).

**Lưu ý quan trọng**: tổng số box sau khi rà là 219, khác với `n_train_boxes = 177` mà
`metrics_round1.json` ghi nhận đã dùng để train. Chênh lệch ~42 box gợi ý model vòng 1 có thể đã
được train trên một phiên bản nhãn **chưa rà đầy đủ** (gần 169 box AI gốc hơn), chứ chưa chắc đã
dùng đúng bộ 219 box đã rà kỹ trong CVAT — đây là nghi vấn cần làm rõ với pipeline trước khi kết
luận chắc chắn nguyên nhân AP50 giảm.

**Nhóm xe tốt lên/xấu đi**: tất cả các nhóm đều xấu đi rõ rệt — recall xe nhỏ từ 18.2% → 0%, xe vừa
từ 54.7% → 1.7%, xe lớn từ 56.1% → 4.9%. Không có nhóm nào cải thiện.

Nhìn vào `compare_round1.jpg`, một ca kết quả đổi rõ rệt: ở cả 4 khung tham khảo (frame_0050,
frame_0150, frame_0250, frame_0350), cột "round 1" gần như trống box (ví dụ frame_0050: cold start
TP=11/19 nhưng round 1 TP=0/19; frame_0350: cold start TP=9/23, round 1 TP=0/23) — mô hình sau
fine-tune gần như không vẽ box nào trên các cảnh có mật độ xe cao, dù cold start đã phát hiện được
phần lớn. Đây là dấu hiệu **overfitting / catastrophic forgetting**: chỉ 12 ảnh huấn luyện 50 epoch
không đủ đa dạng để mô hình giữ được kiến thức tổng quát đã học từ COCO. Vì dữ liệu train (sau khi
rà) khá đầy đủ và chính xác (167 giữ nguyên đúng, chỉ 1 lỗi thật sự cần xoá) nên **nguyên nhân sụp
AP50 nhiều khả năng nằm ở quy trình huấn luyện** (learning rate quá cao, không đóng băng backbone,
thiếu ảnh nền/âm tính, hoặc — như lưu ý ở trên — có thể train nhầm trên phiên bản nhãn chưa rà xong)
chứ không phải do chất lượng gán nhãn kém.

Phân biệt ba nguồn thông tin: **Blind Scan** (`BLIND_SCAN.md`) cho thấy quan sát độc lập của mình
trên `frame_0099.jpg` — 21 xe nhìn thấy bằng mắt, đúng bằng 13 box AI cộng 8 xe bị bỏ sót; **Review
Log** (`REVIEW_LOG.csv`) ghi lại đúng 8 hành động "added" đó cùng 1 "removed" (frame_0312) và 1
"edited" (frame_0326) — khớp hoàn toàn với Blind Scan, không có sai lệch giữa quan sát trước và sau;
**round1_diff.md** (kết quả sau khi model học từ các nhãn đã sửa) lại cho thấy model **không hề học
được** các sửa chữa này — recall sụp gần về 0 trên toàn bộ test set, kể cả những kiểu xe tương tự
các xe mình đã thêm tay. Điều này tách bạch rõ: lỗi nằm ở khâu **huấn luyện**, không phải ở khâu
**quan sát** hay **gán nhãn**.

**Một ca khó theo guideline**: `frame_0312.jpg` — AI vẽ một box khá lớn ở vùng phải-giữa khung hình
nhưng không khớp xe thật nào; theo `HUONG_DAN.txt` (xoá box không phải xe, box ôm vệt đèn), đây là
trường hợp cần xoá vì nhiều khả năng là vệt sáng đèn đường hoặc phản chiếu mặt đường bị nhận nhầm
thành xe, chứ không phải một xe thật bị vẽ lệch nhẹ.

## 5. Kết luận và giới hạn

Kết quả vòng 1 **tệ hơn đáng kể** so với cold start: AP50 giảm 35.1 điểm phần trăm (0.771 → 0.420),
recall sụp từ 48.9% xuống 1.7%, dù dữ liệu train đã được rà nhãn cẩn thận (98.8% box AI đúng, chỉ
bổ sung thêm xe bị bỏ sót). Đây là dấu hiệu rõ ràng của overfitting/catastrophic forgetting do tập
train quá nhỏ (12 ảnh) so với độ phức tạp của cảnh giao thông ban đêm, cộng thêm khả năng training
đã dùng nhầm phiên bản nhãn chưa rà đầy đủ (177 so với 219 box).

Vì kết quả giảm mạnh, quyết định hợp lý là **không tiếp tục dùng model vòng 1 làm nền**. Trước khi
train vòng 2 cần: (1) xác nhận lại pipeline đóng gói nhãn có dùng đúng 219 box đã rà trong CVAT hay
không; (2) xem lại learning rate, số epoch, có nên đóng băng backbone; (3) cân nhắc trộn lại dữ liệu
gốc/negative samples để tránh quên kiến thức COCO.

Hai ca còn yếu/bất định đề xuất cho vòng sau: `frame_0372.jpg` (score 0.9101, bị loại chỉ vì cách
`frame_0369.jpg` đã chọn 1.2s — dưới `min_gap_s=2.0`, không phải vì kém giá trị, chi phí rà ước
tính ~30+ box theo `n_boxes` trong CSV) và `frame_0368.jpg` (score 0.9003, tương tự, cách
`frame_0369.jpg` chỉ 0.4s) — cả hai đáng cân nhắc cho vòng sau nếu mở rộng ngân sách, nhưng cần
kiểm tra kỹ nguy cơ gần trùng cảnh với `frame_0369.jpg` đã có trong tập train.

Ba giới hạn của thiết lập hiện tại và ảnh hưởng đến kết luận:

- **Test set chỉ 20 ảnh (403 box)**: mẫu nhỏ khiến các số đo có phương sai cao — cần thận trọng khi
  so sánh chênh lệch nhỏ giữa các vòng.
- **Luật bỏ qua xe quá nhỏ (dưới 16 px)**: số liệu recall không phản ánh khả năng phát hiện xe rất
  nhỏ/rất xa.
- **Nhãn tham chiếu do mô hình tạo, chưa được rà thủ công**: một phần FP/FN có thể là lỗi của chính
  nhãn tham chiếu — cần đối chiếu Blind Scan trước khi kết luận mô hình sai ở ca cụ thể nào. Thêm
  vào đó, chênh lệch 177 vs 219 box (mục 4) cho thấy khả năng có lỗi ở khâu chuẩn bị dữ liệu train,
  cần xác minh trước khi đổ hoàn toàn lỗi cho lựa chọn mẫu hay chất lượng gán nhãn.

Nếu AP50 giảm (như đã xảy ra ở vòng 1), trước khi train thêm cần kiểm tra theo đúng thứ tự: (1) dữ
liệu đưa vào train có đúng là bản đã rà đầy đủ hay không (179 vs 219 box); (2) learning rate/epoch
có gây overfit không; (3) tập train có lệch phân bố so với test không; (4) nên train tiếp từ
checkpoint cold start hay từ checkpoint vòng 1 đã hỏng.
